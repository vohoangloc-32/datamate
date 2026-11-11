"""
accounting_cleaner.py — Rulebook-Compliant Accounting Hygiene & Status Filtering

Implements:
1. Canonical 4-class order status classification (Rulebook §8 line 110:
   completed / cancel / return / shipping) without substring false positives
   (e.g., 'hoàn' never matches 'Hoàn thành').
2. Structural subtotal/footer row removal and repeated multi-sheet subheader
   removal (safe against surname 'Tống' and product 'Công tắc').
3. Exact-duplicate row deduplication on order-keyed tables only
   (Rulebook §8 lines 111–112).
4. Metric-aware order status filtering (Rulebook §1, §4, §5, §6) with
   whole-order exclusion and cross-table propagation to detail tables.
5. Safe walk-in customer ('khách lẻ') exclusion on customer ranking questions
   (Rulebook §3 line 46), preserving customer tier columns ('Nhóm khách hàng').
6. Rulebook §1 revenue column normalization ('Tổng tiền hàng' vs
   'Tổng sau giảm giá') and Rulebook §3 average-order-value VND rounding.
"""

import calendar
import re
import unicodedata
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


def normalize_text(value: Any) -> str:
    """Fold Vietnamese diacritics and case for keyword matching."""
    text = unicodedata.normalize("NFKD", str(value).casefold())
    text = "".join(c for c in text if not unicodedata.combining(c)).replace("đ", "d")
    return re.sub(r"\s+", " ", text).strip()


def compact_identifier(value: Any) -> str:
    """Alphanumeric-only identifier for column name matching."""
    return re.sub(r"[^a-z0-9]", "", normalize_text(value))


# ---------------------------------------------------------------------------
# 1. Canonical 4-class order status classification (Rulebook §8 line 110)
# ---------------------------------------------------------------------------

# Order matters: bare 'hoan' is never matched so 'Hoàn thành' is never confused
# with 'Hoàn trả' / 'Đã hoàn hàng'.
_STATUS_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    (
        "return",
        re.compile(
            r"\b(tra hang|doi tra|tra lai|hang tra ve|hoan tra|hoan hang|hoan tien|returned?|refund\w*)\b"
        ),
    ),
    (
        "cancel",
        re.compile(
            r"\b(huy|huy don|cancel\w*|khong giao duoc|giao that bai|failed)\b"
        ),
    ),
    (
        "shipping",
        re.compile(
            r"\b(dang giao|dang van chuyen|dang xu ly|dang gui|cho giao|cho xu ly|shipping|pending|dang giao dich)\b"
        ),
    ),
    (
        "completed",
        re.compile(
            r"\b(hoan thanh|xong|da xong|thanh cong|da lay|da giao|ok|completed?|delivered|done)\b"
        ),
    ),
)


def classify_status(value: Any) -> Optional[str]:
    """Map a raw cell value to one of {'completed', 'cancel', 'return', 'shipping'} or None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    n = normalize_text(value)
    if not n:
        return None
    for label, pattern in _STATUS_PATTERNS:
        if pattern.search(n):
            return label
    return None


def detect_status_column(df: pd.DataFrame) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    Detect the order-status column via cell-value profiling.

    Returns (column_name, {raw_value_str: canonical_class}) if a column has
    >=90% of its non-null values classified into the 4 canonical classes,
    including at least 'completed' and one of {'cancel', 'return'}.
    """
    best: Optional[Tuple[float, str, Dict[str, str]]] = None
    for col in df.columns:
        if str(col).startswith("__"):
            continue
        series = df[col]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            continue
        vc = series.dropna().astype(str).value_counts()
        if vc.empty or len(vc) > 40:
            continue
        mapping = {val: classify_status(val) for val in vc.index}
        classified_rows = sum(cnt for val, cnt in vc.items() if mapping[val])
        share = classified_rows / vc.sum()
        classes = {cls for cls in mapping.values() if cls}
        if share >= 0.90 and "completed" in classes and (classes & {"cancel", "return"}):
            clean_map = {val: cls for val, cls in mapping.items() if cls}
            if best is None or share > best[0]:
                best = (share, col, clean_map)
    return (best[1], best[2]) if best else None


# ---------------------------------------------------------------------------
# 2. Order ID detection & structural trap removal (subtotals, subheaders, dups)
# ---------------------------------------------------------------------------

_ORDER_ID_TOKENS: Tuple[str, ...] = (
    "madon",
    "mahd",
    "sohd",
    "mahoadon",
    "sohoadon",
    "orderid",
    "orderno",
    "ordernumber",
    "invoice",
    "machungtu",
    "sochungtu",
    "soct",
    "madh",
    "sodon",
    "iddon",
    "donhangid",
    "mabill",
    "sophieu",
    "maphieu",
    "magd",
    "magiaodich",
    "madonhang",
)

# Strictly accented Vietnamese NFC subtotal/footer patterns so customer
# surnames like 'Tống' and products like 'Công tắc' are never matched.
_SUBTOTAL_NFC_RE = re.compile(
    r"^\s*(?:"
    r"tổng(?:\s+cộng|\s+ngày|\s+tháng|\s+quý|\s+năm|\s+kết|\s*:)?|"
    r"cộng(?:\s+ngày|\s+tháng|\s+quý|\s+năm|\s*:)?|"
    r"total|subtotal|ghi chú\s*:"
    r")(?:\s+\d|\s*$)",
    re.IGNORECASE,
)

_SUBHEADER_TOKENS: Set[str] = {
    "status",
    "trị giá",
    "mã hóa đơn",
    "mã phiếu",
    "ghi chú tt",
    "ngày giao dịch",
    "tên sp",
    "số lượng hàng",
    "giá lẻ",
    "unit",
    "donvi",
    "tenkh",
}


def find_order_id_column(df: pd.DataFrame) -> Optional[str]:
    """Return the order-ID column name if the table is keyed by order."""
    for col in df.columns:
        if any(token in compact_identifier(col) for token in _ORDER_ID_TOKENS):
            return col
    if not df.empty:
        row0 = df.iloc[0]
        for col in df.columns:
            val = row0[col]
            if isinstance(val, str) and any(
                token in compact_identifier(val) for token in _ORDER_ID_TOKENS
            ):
                return col
    return None


def remove_subheader_and_subtotal_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove repeated sub-header rows and subtotal/footer rows before ffill.

    A row is treated as a subtotal/footer only if:
    - At least 2 data columns in the row are NaN/NaT, AND
    - At least one string cell matches `_SUBTOTAL_NFC_RE` with Vietnamese accents preserved.
    """
    if df.empty:
        return df

    work = df
    data_cols = [c for c in work.columns if not str(c).startswith("__source_")]
    if not data_cols:
        return work

    # 1. Repeated sub-headers in sheets with 'Unnamed:' columns
    if any(str(c).startswith("Unnamed:") for c in work.columns):
        first_col = data_cols[0]
        s0 = work[first_col].dropna().astype(str).str.strip().str.casefold()
        cand_idx = s0[s0.isin({"hàng", "đơn", "status", "mã phiếu", "mã hóa đơn"})].index
        if len(cand_idx) > 0:
            drop_hdr = []
            for idx in cand_idx:
                row_vals = [
                    unicodedata.normalize("NFC", str(v)).strip().casefold()
                    for v in work.loc[idx, data_cols].values
                    if pd.notna(v)
                ]
                if sum(1 for v in row_vals if v in _SUBHEADER_TOKENS) >= 2:
                    drop_hdr.append(idx)
            if drop_hdr:
                work = work.drop(index=drop_hdr).reset_index(drop=True)

    # 2. Structural subtotal / footer rows
    if len(data_cols) >= 3 and not work.empty:
        na_counts = work[data_cols].isna().sum(axis=1)
        cand_sub = work.index[na_counts >= 2]
        if len(cand_sub) > 0:
            sub_mask = pd.Series(False, index=cand_sub)
            for col in data_cols:
                series = work[col]
                if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                    s_col = work.loc[cand_sub, col].dropna().astype(str)
                    if not s_col.empty:
                        s_nfc = s_col.map(lambda v: unicodedata.normalize("NFC", v).strip())
                        matched = s_nfc.str.contains(_SUBTOTAL_NFC_RE, regex=True, na=False)
                        sub_mask.loc[matched[matched].index] = True
            if sub_mask.any():
                work = work.drop(index=sub_mask[sub_mask].index).reset_index(drop=True)

    return work


def dedup_order_table_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rulebook §8 lines 111–112:
    Deduplicate 100% identical rows ONLY when the table records by order ID.
    Tables without an order ID column keep identical rows.
    """
    if df.empty:
        return df
    oid = find_order_id_column(df)
    if oid is None:
        return df
    data_cols = [c for c in df.columns if not str(c).startswith("__source_")]
    if not data_cols:
        return df
    if df.duplicated(subset=data_cols).any():
        return df.drop_duplicates(subset=data_cols).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. Question intent classifiers for Rulebook metric-aware rules
# ---------------------------------------------------------------------------

_NEEDS_ALL_STATUS_RE = re.compile(
    r"\b(huy|tra hang|tra lai|hoan|no|cong no|gia ban|dot giao|dot hang|luot giao|chuyen giao|ti le|ty le)\b"
    r"|\b(?:xe\s+(?:da\s+)?giao|xe\s+cho|giao\s+(?:\w+\s+){0,3}dot)\b"
)
_RANK_CUSTOMER_RE = re.compile(
    r"\b(khach hang|khach nao|ai mua|ai la khach|khach mua)\b.*\b(nhieu nhat|cao nhat|top|xep hang|nhieu don|nhiu don|nhieu lan|nhiu lan|nhieu tien|nhiu tien)\b"
    r"|\b(top|nhieu nhat|cao nhat|liet ke \d+|bang \d+)\b.*\b(khach)\b"
    r"|\b(?:\d+|top)\s+khach\b.*\b(?:cao|nhieu|lon)\s+nhat\b"
    r"|\bkhach\b.*\b(?:tong\s+tien|chi\s+tieu|doanh\s+thu|so\s+don|so\s+lan)\b.*\b(?:cao|nhieu|lon)\s+nhat\b"
    r"|\b(khach|khach hang)\b.*\bmua tu \d+ (?:don|lan|bill)\b"
    r"|\b(?:bao nhieu|bnhiu|bn|may|so) khach\b.*\b(?:da mua|mua hang|mua tu)\b"
)
_AOV_RE = re.compile(
    r"\b(trung binh moi (?:don|bill|order)|binh quan moi (?:don|bill|order)|gia tri trung binh mot (?:don|bill|order)"
    r"|doanh thu trung binh.*(?:don|bill|order)|dthu trung binh.*(?:don|bill|order)|doanh thu binh quan.*(?:don|bill|order)|dthu binh quan.*(?:don|bill|order))\b"
)
_PURE_REVENUE_RE = re.compile(
    r"\b(doanh thu|dthu|revenue|ban (?:dc|duoc).*?(?:tien|bao nhieu|bn|nhiu)|thu (?:ve|duoc).*?tien|tong tien ban hang|tien ban hang|tien hang|ban.*bao nhieu tien)\b"
)
_MONETARY_AVG_RE = re.compile(
    r"\b(gia nhap binh quan|nhap vao gia binh quan|gia ban binh quan|don gia binh quan"
    r"|gia tri trung binh|dthu binh quan|doanh thu binh quan|tien trung binh"
    r"|doanh thu trung binh|dthu trung binh|chi phi trung binh|trung binh moi don"
    r"|binh quan moi don|gia binh quan|trung binh.*tien|binh quan.*tien|trung binh moi dot|binh quan moi dot)\b"
)
_DEBT_OR_PAID_RE = re.compile(
    r"\b(no|cong no|da tra|da thu|thuc thu|con no|phai thu|tra bot no|thu no)\b"
)
_SPECIFIC_CANCEL_REASON_RE = re.compile(
    r"\b(vi|ly do|li do|nguyen nhan|giao tre|het hang|doi y)\b"
)

_WALKIN_VALUES: Set[str] = {
    "khach le",
    "kl",
    "le",
    "vang lai",
    "khach vang lai",
    "khach mua le",
}


def _normalize_question(question: str) -> str:
    """Normalize question slang/Telex typos and fold Vietnamese diacritics."""
    try:
        from .question_understanding import normalize_question_slang
        question = normalize_question_slang(question)
    except Exception:
        pass
    return normalize_text(question)


def question_needs_all_statuses(question: str) -> bool:
    """True when the question asks about cancelled/returned orders, rates, debt, selling price, or delivery batches."""
    return bool(_NEEDS_ALL_STATUS_RE.search(_normalize_question(question)))


def question_is_customer_ranking(question: str) -> bool:
    """True when the question ranks or finds top individual customers or counts repeat customers."""
    return bool(_RANK_CUSTOMER_RE.search(_normalize_question(question)))


def question_is_average_order_value(question: str) -> bool:
    """True when the question asks for average value/revenue per order (Rulebook §3 line 48)."""
    return bool(_AOV_RE.search(_normalize_question(question)))


def question_is_monetary_average(question: str) -> bool:
    """True when question asks for average/mean of monetary values (Rulebook §9 line 133)."""
    q = _normalize_question(question)
    return bool(_MONETARY_AVG_RE.search(q)) and not bool(re.search(r"\b(ti le|ty le|phan tram|%|so luong|so don|so ngay)\b", q))


def question_is_pure_revenue(question: str) -> bool:
    """True when the question asks for revenue and not debt/paid amounts."""
    q = _normalize_question(question)
    return bool(_PURE_REVENUE_RE.search(q)) and not bool(_DEBT_OR_PAID_RE.search(q))


def round_half_up_vnd(value: Any) -> Any:
    """Round numeric VND amounts to nearest integer with half rounded up (Rulebook §3 line 48)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    try:
        d = Decimal(str(float(value)))
        return int(d.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    except Exception:
        return value


# ---------------------------------------------------------------------------
# 4. Question-scoped table preparation & Rulebook code patching
# ---------------------------------------------------------------------------

def prepare_tables_for_question(
    tables: Dict[str, pd.DataFrame],
    question: str = "",
) -> Dict[str, pd.DataFrame]:
    """
    Return a question-scoped dictionary of DataFrames with:
    1. Structural subheaders/subtotals removed and order-keyed exact duplicates removed.
    2. Cancelled and returned orders removed (whole-order level + cross-table propagation)
       UNLESS `question_needs_all_statuses(question)` is True.
    3. Walk-in customer labels removed on customer-name columns (excluding tier columns
       like 'Nhóm khách hàng') when `question_is_customer_ranking(question)` is True.
    """
    filter_status = not question_needs_all_statuses(question) if question else True
    filter_walkin = question_is_customer_ranking(question) if question else False

    out: Dict[str, pd.DataFrame] = {}
    bad_order_ids: Set[Any] = set()

    # Detect standalone return tables (e.g. tra_hang) and collect their order IDs
    for name, df in tables.items():
        if any(term in name.lower() for term in ("tra_hang", "phieu_tra", "hoan_tra", "return")):
            oid = find_order_id_column(df)
            if oid is not None:
                returned_ids = set(df[oid].dropna().unique())
                if filter_status:
                    bad_order_ids.update(returned_ids)
        # Rulebook §1 line 10: detect return quantity columns (e.g. BanHang_BanTheoLo, DonHang_Lo)
        for col in df.columns:
            c_comp = compact_identifier(col)
            if any(k in c_comp for k in ("sltra", "soluongtra", "sltradvtcoban", "soluongtradvtcoban")):
                s_numeric = pd.to_numeric(df[col], errors="coerce").fillna(0)
                ret_mask = s_numeric > 0
                if ret_mask.any():
                    oid = find_order_id_column(df)
                    if oid is not None:
                        returned_ids = set(df.loc[ret_mask, oid].dropna().unique())
                        if filter_status:
                            bad_order_ids.update(returned_ids)
            # Rulebook §1: detect negative quantity in sales lines (handwritten ledger return records)
            elif any(k in c_comp for k in ("soluong", "sl", "qty", "quantity", "soluonghang")) and not any(k in c_comp for k in ("tra", "quy", "ton", "nhap")):
                s_numeric = pd.to_numeric(df[col], errors="coerce").fillna(0)
                neg_mask = s_numeric < 0
                if neg_mask.any():
                    oid = find_order_id_column(df)
                    if oid is not None:
                        neg_ids = set(df.loc[neg_mask, oid].dropna().unique())
                        if filter_status:
                            bad_order_ids.update(neg_ids)

    for name, df in tables.items():
        work = remove_subheader_and_subtotal_rows(df.copy())
        work = dedup_order_table_rows(work)
        data_cols = [c for c in work.columns if not str(c).startswith("__source_")]
        oid = find_order_id_column(work)

        det = detect_status_column(work)
        if det and filter_status:
            status_col, mapping = det
            cls = work[status_col].map(mapping)
            if oid is not None:
                keys = work[oid].ffill()
                bads = set(keys[cls.isin(["cancel", "return"])].dropna().unique())
                bad_order_ids.update(bads)
                work = work.loc[~keys.isin(bads)].copy()
            else:
                work = work.loc[~cls.isin(["cancel", "return"])].copy()

        if filter_walkin:
            for col in data_cols:
                c_comp = compact_identifier(col)
                if (
                    any(t in c_comp for t in ("tenkhach", "khachhang", "benmua", "tenkh"))
                    or c_comp in ("khach", "kh", "khachmua", "nguoimua")
                ) and not any(x in c_comp for x in ("nhom", "loai", "ma", "so")):
                    s_col = work[col].dropna().astype(str)
                    vc_map = {
                        v: normalize_text(v) in _WALKIN_VALUES for v in s_col.unique()
                    }
                    walkin_mask = work[col].map(vc_map).fillna(True).astype(bool)
                    if walkin_mask.any():
                        work = work.loc[~walkin_mask].copy()
                elif c_comp in ("makh", "makhach", "makhachhang", "customerid"):
                    id_walkin = (
                        work[col].isna()
                        | work[col].astype(str).str.strip().map(
                            lambda v: not v or normalize_text(v) in _WALKIN_VALUES
                        )
                    ).astype(bool)
                    if id_walkin.any() and not id_walkin.all():
                        work = work.loc[~id_walkin].copy()

        # Rulebook §6 VLXD: deduplicate (order_id, trip_col) directly when counting delivery trips
        if question:
            q_norm_prep = _normalize_question(question)
            if any(k in q_norm_prep for k in ("dot giao", "luot giao", "xe da giao", "may dot", "bao nhieu dot", "so dot", "chuyen giao")) and not any(
                bool(re.search(rf"\b{u}\b", q_norm_prep)) for u in ("kg", "tan", "m3", "vien", "so luong")
            ) and not bool(re.search(r"\bbao\s+(?:xi|cat|da|gao)\b", q_norm_prep)):
                trip_col = next((c for c in work.columns if compact_identifier(c) in ("luotgiao", "langiao", "dotgiao")), None)
                if trip_col and oid:
                    work = work.drop_duplicates(subset=[oid, trip_col]).reset_index(drop=True)

        out[name] = work

    # Propagate excluded order IDs across all tables sharing any order ID column
    if filter_status and bad_order_ids:
        bad_id_strs = {str(x).strip() for x in bad_order_ids if pd.notna(x)}
        for name, work in list(out.items()):
            oid_cols = [c for c in work.columns if any(token in compact_identifier(c) for token in _ORDER_ID_TOKENS)]
            for col in oid_cols:
                drop = work[col].astype(str).str.strip().isin(bad_id_strs)
                if drop.any():
                    work = work.loc[~drop].copy()
            out[name] = work

    # Rulebook §1: Sync item line sums from detail tables into header tables
    detail_amount_cols = ("T.Tiền", "Thành tiền", "Amount", "Tiền bán", "Thanh tien", "ThanhTien")
    for hname, hdf in list(out.items()):
        h_oid = find_order_id_column(hdf)
        if h_oid is None or hdf.empty:
            continue
        h_orders = set(hdf[h_oid].dropna().unique())
        if not h_orders:
            continue
        for dname, ddf in out.items():
            if dname == hname or ddf.empty:
                continue
            d_oid = find_order_id_column(ddf)
            if d_oid is None:
                continue
            d_orders = set(ddf[d_oid].dropna().unique())
            if len(d_orders.intersection(h_orders)) > 0.5 * len(d_orders):
                line_col = next((c for c in detail_amount_cols if c in ddf.columns), None)
                if line_col and "Tổng tiền hàng" not in hdf.columns and "Tiền hàng trước giảm" not in hdf.columns:
                    line_sums = ddf.groupby(d_oid)[line_col].sum()
                    hdf["Tổng tiền hàng"] = hdf[h_oid].map(line_sums).fillna(0)
                    out[hname] = hdf
                break

    # Auto-map Customer Names into order tables that only have Customer IDs
    for hname, hdf in list(out.items()):
        h_oid = find_order_id_column(hdf)
        if h_oid is None or hdf.empty:
            continue
        cust_id_col = next((c for c in hdf.columns if compact_identifier(c) in ("makh", "makhach", "makhachhang", "makhachmua", "customerid", "sokh")), None)
        cust_name_col = next((c for c in hdf.columns if compact_identifier(c) in ("tenkh", "tenkhach", "tenkhachhang", "customername")), None)
        if cust_id_col and not cust_name_col:
            for cname, cdf in out.items():
                if cname == hname or cdf.empty:
                    continue
                c_id_col = next((c for c in cdf.columns if compact_identifier(c) in ("makh", "makhach", "makhachhang", "makhachmua", "customerid", "sokh", "makh")), None)
                c_name_col = next((c for c in cdf.columns if compact_identifier(c) in ("tenkh", "tenkhach", "tenkhachhang", "customername", "hovaten", "kh", "khach", "khachhang", "khachlesi", "khachle") and c != c_id_col), None)
                if c_id_col and c_name_col:
                    id_map = dict(zip(cdf[c_id_col], cdf[c_name_col]))
                    hdf["Tên khách hàng"] = hdf[cust_id_col].map(id_map).fillna(hdf[cust_id_col])
                    out[hname] = hdf
                    break

    return out


_REDUNDANT_STATUS_COLS_PATTERN = (
    r"(?:Trạng thái|Trạng thái đơn hàng|Trạng thái xử lý|Trạng thái đơn|"
    r"Tình trạng|Tình trạng đơn|Kết quả|Xử lý|Status|Ghi chú TT|trang_thai|"
    r"TinhTrang|Unnamed: [45])"
)


def _fix_date_boundaries(code: str) -> str:
    """
    Ensure date range comparisons don't inadvertently cut off timestamps within the boundary day.
    1. Rewrite `<= pd.Timestamp('YYYY-MM-DD')` to `< (pd.Timestamp('YYYY-MM-DD') + pd.Timedelta(days=1))`
    2. Rewrite `< pd.Timestamp('YYYY-MM-last_day')` (month-end) to `< (pd.Timestamp('YYYY-MM-last_day') + pd.Timedelta(days=1))`
    """
    if not code:
        return code

    def replace_lte(match: re.Match) -> str:
        date_str = match.group(1)
        return f"< (pd.Timestamp('{date_str}') + pd.Timedelta(days=1))"

    # Matches <= pd.Timestamp('YYYY-MM-DD') or <= pd.to_datetime('YYYY-MM-DD')
    res = re.sub(
        r"<=\s*(?:pd\.Timestamp|pd\.to_datetime)\(['\"](\d{4}-\d{2}-\d{2})['\"]\)",
        replace_lte,
        code,
    )
    # Matches <= 'YYYY-MM-DD' when preceded by a comparison operator context
    res = re.sub(
        r"(?<=[^\w])<=\s*['\"](\d{4}-\d{2}-\d{2})['\"]",
        replace_lte,
        res,
    )

    def replace_lt_month_end(match: re.Match) -> str:
        date_str = match.group(1)
        parts = date_str.split("-")
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        try:
            _, last_day = calendar.monthrange(y, m)
            if d == last_day:
                return f"< (pd.Timestamp('{date_str}') + pd.Timedelta(days=1))"
        except Exception:
            pass
        return match.group(0)

    res = re.sub(
        r"<\s*(?:pd\.Timestamp|pd\.to_datetime)\(['\"](\d{4}-\d{2}-(?:28|29|30|31))['\"]\)",
        replace_lt_month_end,
        res,
    )
    res = re.sub(
        r"(?<=[^\w])<\s*['\"](\d{4}-\d{2}-(?:28|29|30|31))['\"]",
        replace_lt_month_end,
        res,
    )
    return res


def patch_code_and_tables_for_rulebook(
    code: str,
    question: str,
    tables: Dict[str, pd.DataFrame],
) -> Tuple[str, Dict[str, pd.DataFrame]]:
    """
    Apply Rulebook §1 & §8 post-processing to LLM code and execution tables:
    1. When tables already have cancel/return orders removed (`not needs_all`),
       neutralize single-synonym status filters like `df['Trạng thái'] == 'Hoàn thành'`
       so valid 'Đang giao' and synonym ('xong', 'đã lấy', 'ok') orders are not lost.
    2. When the question asks about cancelled orders (without an unanswerable
       reason clause like 'vì giao trễ'), unify cancel synonyms in the table
       and code so multi-synonym shops ('hủy' + 'Huỷ đơn') are counted completely.
    3. When the question asks for pure revenue (Rulebook §1 line 15), redirect
       post-discount invoice columns ('Tổng sau giảm giá', 'TongTien') to
       pre-discount item totals ('Tổng tiền hàng', 'Tien hang') when both exist.
    """
    if not question or not code:
        return code, tables

    qtext = _normalize_question(question)
    needs_all = question_needs_all_statuses(question)
    patched = code
    exec_tables = tables

    if not needs_all:
        # Neutralize redundant status filters (including ~df['Trạng thái'].isin(...)) on pre-filtered tables
        patched = re.sub(
            rf"~?\s*([A-Za-z0-9_]+\[\s*['\"]{_REDUNDANT_STATUS_COLS_PATTERN}['\"]\s*\])\.isin\(\[[^\]]*\]\)",
            r"\1.notna()",
            patched,
        )
        patched = re.sub(
            rf"~?\s*([A-Za-z0-9_]+\[\s*['\"]{_REDUNDANT_STATUS_COLS_PATTERN}['\"]\s*\])\s*(?:==|!=)\s*['\"][^'\"]*['\"]",
            r"\1.notna()",
            patched,
        )
    elif "huy" in qtext and not _SPECIFIC_CANCEL_REASON_RE.search(qtext):
        exec_tables = {k: v.copy() for k, v in tables.items()}
        for tname, df in list(exec_tables.items()):
            det = detect_status_column(df)
            if det:
                scol, mp = det
                c_vals = [v for v, c in mp.items() if c == "cancel"]
                r_vals = [v for v, c in mp.items() if c == "return"]
                if c_vals:
                    # Relabel return values to __RETURN_ONLY__ so pure cancel questions
                    # matching .isin(['Đã hủy', 'Đã trả hàng']) will ONLY match cancelled rows.
                    if r_vals:
                        df[scol] = df[scol].map(
                            lambda v: "__RETURN_ONLY__" if mp.get(str(v)) == "return" else v
                        )
                    df[scol] = df[scol].map(
                        lambda v: "__CANCEL__" if mp.get(str(v)) == "cancel" else v
                    )
                    for cv in c_vals + [
                        "hủy",
                        "huỷ",
                        "Đã hủy",
                        "Đã huỷ",
                        "hủy đơn",
                        "Huỷ đơn",
                        "Đã hủy đơn",
                    ]:
                        patched = patched.replace(f"'{cv}'", "'__CANCEL__'").replace(
                            f'"{cv}"', '"__CANCEL__"'
                        )

    # Multi-line order deduplication for order count & cancellation rate
    is_order_counting = any(
        k in qtext for k in ("dem so don", "may don", "bao nhieu don", "so luong don", "may bill", "bao nhieu bill", "ti le", "ty le", "% don", "so don", "ti le huy", "tỷ lệ hủy")
    )
    if is_order_counting:
        primary_oid = None
        for tname, df in exec_tables.items():
            oid = find_order_id_column(df)
            if oid is not None and df[oid].duplicated().any():
                if primary_oid is None:
                    primary_oid = oid
                patched = re.sub(
                    rf"(\b{tname}\[[^\]]*\])\.shape\[0\]",
                    rf"\1.drop_duplicates(subset=['{oid}']).shape[0]",
                    patched,
                )
                patched = re.sub(
                    rf"(\b{tname})\.shape\[0\]",
                    rf"\1['{oid}'].nunique()",
                    patched,
                )
                patched = re.sub(
                    rf"len\(\s*(\b{tname}\[[^\]]*\])\s*\)",
                    rf"\1.drop_duplicates(subset=['{oid}']).shape[0]",
                    patched,
                )
                patched = re.sub(
                    rf"len\(\s*(\b{tname})\s*\)",
                    rf"\1['{oid}'].nunique()",
                    patched,
                )
                # Also rewrite order cancellation calculations matching __CANCEL__
                patched = re.sub(
                    rf"(\w+\[[^\]]*__CANCEL__[^\]]*\])\.shape\[0\]",
                    rf"\1.drop_duplicates(subset=['{oid}']).shape[0]",
                    patched,
                )
        # Also catch intermediate filtered DataFrame variables (e.g. result = len(df_q1) or df_q1.shape[0])
        if primary_oid and ".nunique(" not in patched and ".drop_duplicates(" not in patched and not any(
            w in qtext for w in ("mat hang", "san pham", "khach", "ngay", "dot")
        ):
            patched = re.sub(
                r"result\s*=\s*len\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)",
                rf"result = (\1['{primary_oid}'].nunique() if isinstance(\1, pd.DataFrame) and '{primary_oid}' in \1.columns else len(\1))",
                patched,
            )
            patched = re.sub(
                r"result\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\.shape\[0\]",
                rf"result = (\1['{primary_oid}'].nunique() if isinstance(\1, pd.DataFrame) and '{primary_oid}' in \1.columns else \1.shape[0])",
                patched,
            )

    # Average Order Value (AOV) on multi-line tables:
    if question_is_average_order_value(question):
        for line_col in ("Thành tiền", "Tiền bán", "Thành tiền trước giảm", "Số tiền", "ThanhTien", "thanh_tien", "TienBan", "Tổng tiền", "TongTien"):
            # 1. Direct series mean without groupby on multi-line tables:
            # df['Tổng tiền'].mean() -> df.groupby(oid)['Tổng tiền'].sum().mean()
            for tname, df in exec_tables.items():
                oid = find_order_id_column(df)
                if oid is not None and df[oid].duplicated().any():
                    patched = re.sub(
                        rf"(\b{tname}\[[^\]]*\])\[\s*['\"]{line_col}['\"]\s*\]\.mean\(\)",
                        rf"\1.groupby('{oid}')['{line_col}'].sum().mean()",
                        patched,
                    )
            # 2. Grouped mean: .groupby(oid)[line_col].mean() -> .groupby(oid)[line_col].sum()
            patched = re.sub(
                rf"\.groupby\(\s*(['\"][^'\"]+['\"])\s*\)\[\s*['\"]{line_col}['\"]\s*\]\.mean\(\)",
                rf".groupby(\1)['{line_col}'].sum()",
                patched,
            )

    # Relational Fan-out Guard:
    # When code merges tables and calculates .sum() on order header total columns (e.g. 'Tổng tiền hàng'),
    # rewrite to line item total (e.g. 'Thành tiền') on the merged table variable.
    has_merge = bool(re.search(r"\b(?:merge|join)\b", patched))
    if has_merge:
        has_tong_tien = any("Tổng tiền hàng" in df.columns for df in tables.values())
        has_thanh_tien = any("Thành tiền" in df.columns for df in tables.values())
        if has_tong_tien and has_thanh_tien:
            patched = re.sub(
                r"([A-Za-z0-9_]+(?:\[[^\]]*\])?)\[\s*['\"]Tổng tiền hàng['\"]\s*\]\.sum\(\)",
                r"\1['Thành tiền'].sum()",
                patched,
            )

    # Date boundary guard: <= pd.Timestamp('YYYY-MM-DD') cut off timestamps on that day
    patched = _fix_date_boundaries(patched)

    # Clamp non-leap year Feb 29 (e.g. 2025-02-29 -> 2025-02-28)
    def _clamp_feb29(m):
        yr = int(m.group(1))
        sep = m.group("sep")
        if yr % 4 != 0 or (yr % 100 == 0 and yr % 400 != 0):
            return f"{yr}{sep}02{sep}28"
        return m.group(0)
    patched = re.sub(r"\b(20\d{2})(?P<sep>[/-])02(?P=sep)29\b", _clamp_feb29, patched)

    # Delivery trips deduplication (Rulebook §6 VLXD: each delivery trip of an order counts as one trip)
    if any(k in qtext for k in ("dot giao", "luot giao", "xe da giao", "chuyen giao")):
        for tname, df in exec_tables.items():
            trip_col = next((c for c in df.columns if compact_identifier(c) in ("luotgiao", "langiao", "dotgiao")), None)
            oid = find_order_id_column(df)
            if trip_col and oid:
                patched = re.sub(
                    rf"(\b{tname}(?:\[.+?\])?)\.shape\[0\]",
                    rf"\1.drop_duplicates(subset=['{oid}', '{trip_col}']).shape[0]",
                    patched,
                )
                patched = re.sub(
                    rf"len\(\s*(\b{tname}(?:\[.+?\])?)\s*\)",
                    rf"\1.drop_duplicates(subset=['{oid}', '{trip_col}']).shape[0]",
                    patched,
                )

    # Rulebook §6 line 82: Size M, size L must end with (M) or (L), case-insensitive
    if bool(re.search(r"\bsize\s*m\b", qtext)):
        patched = re.sub(
            r"\.str\.(?:contains|endswith)\(\s*['\"](?:\(?M\)?|\^?.*M.*)['\"]\s*(?:,\s*case\s*=\s*(?:False|True))?\s*\)",
            lambda _: ".str.contains(r'\\(M\\)\\s*$', case=False, regex=True)",
            patched,
        )
    if bool(re.search(r"\bsize\s*l\b", qtext)):
        patched = re.sub(
            r"\.str\.(?:contains|endswith)\(\s*['\"](?:\(?L\)?|\^?.*L.*)['\"]\s*(?:,\s*case\s*=\s*(?:False|True))?\s*\)",
            lambda _: ".str.contains(r'\\(L\\)\\s*$', case=False, regex=True)",
            patched,
        )

    # Rulebook §5 line 63: Drop cancelled import slips
    if any(k in qtext for k in ("tien nhap hang", "tien nhap", "chi phi nhap", "tong tien hang nhap")):
        for tname, df in exec_tables.items():
            if any(w in compact_identifier(tname) for w in ("nhaphang", "phieunhap", "import")):
                status_col = next((c for c in df.columns if any(w in compact_identifier(c) for w in ("trangthai", "tinhtrang", "status"))), None)
                if status_col:
                    patched = re.sub(
                        rf"(\b{tname}\b)(?!\[~?\1\['{status_col}'\])",
                        rf"(\1[~\1['{status_col}'].astype(str).str.lower().str.contains('huy|cancel', na=False)])",
                        patched,
                    )

    # Rulebook §3: Protect against inner join on conversion tables dropping products without large units
    for conv_token in ("quydoi", "quy_doi", "don_vi_sp", "donvisp", "danhmuc", "danh_muc"):
        patched = re.sub(
            rf"\.merge\(\s*([A-Za-z0-9_]*{conv_token}[A-Za-z0-9_]*)\s*,\s*how\s*=\s*['\"]inner['\"]",
            r".merge(\1, how='left'",
            patched,
            flags=re.IGNORECASE,
        )
        patched = re.sub(
            rf"\.merge\(\s*([A-Za-z0-9_]*{conv_token}[A-Za-z0-9_]*)\s*,\s*(?!how\s*=)",
            r".merge(\1, how='left', ",
            patched,
            flags=re.IGNORECASE,
        )
        patched = re.sub(
            rf"\.merge\(\s*([A-Za-z0-9_]*{conv_token}[A-Za-z0-9_]*)\s*\)",
            r".merge(\1, how='left')",
            patched,
            flags=re.IGNORECASE,
        )
        patched = re.sub(
            rf"pd\.merge\(\s*([^,]+),\s*([A-Za-z0-9_]*{conv_token}[A-Za-z0-9_]*)\s*,\s*how\s*=\s*['\"]inner['\"]",
            r"pd.merge(\1, \2, how='left'",
            patched,
            flags=re.IGNORECASE,
        )
        patched = re.sub(
            rf"pd\.merge\(\s*([^,]+),\s*([A-Za-z0-9_]*{conv_token}[A-Za-z0-9_]*)\s*,\s*(?!how\s*=)",
            r"pd.merge(\1, \2, how='left', ",
            patched,
            flags=re.IGNORECASE,
        )
        patched = re.sub(
            rf"pd\.merge\(\s*([^,]+),\s*([A-Za-z0-9_]*{conv_token}[A-Za-z0-9_]*)\s*\)",
            r"pd.merge(\1, \2, how='left')",
            patched,
            flags=re.IGNORECASE,
        )

    # Rulebook §3 lines 43-44: Non-converted items have ratio = 1 (fillna(1) on conversion ratios)
    ratio_cols = ("Tỉ lệ quy đổi", "Ti le quy doi", "Hệ số", "He so", "he_so", "ti_le", "ty_le", "ratio", "Tỉ lệ")
    for rcol in ratio_cols:
        patched = re.sub(
            rf"(\b[A-Za-z0-9_]*\[\s*['\"]{rcol}['\"]\s*\])(?!\s*\.fillna)",
            rf"\1.fillna(1)",
            patched,
        )

    # Rulebook §3 lines 43–44 & §5 line 71: Ensure base-unit quantity ('Số lượng quy đổi') is ready
    has_sl_quy_doi = any("Số lượng quy đổi" in df.columns for df in exec_tables.values())

    # Rulebook §3 lines 43–44: When ranking products (bán chạy nhất, hàng hot nhất), rank by QUANTITY sold unless revenue is specified
    is_product_ranking = any(
        k in qtext
        for k in (
            "ban chay", "chay nhat", "hot nhat", "ban nhieu nhat", "best seller",
            "hang hot", "mon hot", "san pham hot", "top", "mon nao", "san pham nao", "hang nao"
        )
    ) and not any(k in qtext for k in ("doanh thu", "dthu", "tien", "revenue", "gia tri"))

    if has_sl_quy_doi or is_product_ranking:
        if exec_tables is tables:
            exec_tables = {k: v.copy() for k, v in tables.items()}
        for df in exec_tables.values():
            if "Số lượng quy đổi" not in df.columns:
                for raw_q in ("Số lượng", "Số lượng sản phẩm", "so_luong", "SL", "Số lượng hàng", "soluonghang", "SoLuong", "SL lẻ", "Số lượng lẻ", "Quy ra ĐV lẻ", "Số lượng (ĐVT cơ bản)"):
                    if raw_q in df.columns:
                        df["Số lượng quy đổi"] = pd.to_numeric(df[raw_q], errors="coerce").fillna(0)
                        break

    if has_sl_quy_doi:
        # Neutralize LLM manual ratio multiplication when precalculated 'Số lượng quy đổi' exists
        patched = re.sub(
            r"([A-Za-z0-9_]+\[\s*['\"](?:Số lượng quy đổi|Số lượng cơ bản|sl_quy_doi|sl_co_ban|Quantity_in_Base_Unit|Số lượng lẻ)['\"]\s*\]\s*=\s*[A-Za-z0-9_]+\[\s*['\"](?:Số lượng|Số lượng sản phẩm|so_luong|SL|Số lượng quy đổi)['\"]\s*\]\s*\*\s*[A-Za-z0-9_]+\[\s*['\"](?:Tỉ lệ quy đổi|Ti le quy doi|Hệ số|Hệ số quy đổi|He so|ratio)['\"]\s*\](?:\.fillna\([^)]*\))?)",
            r"# \1 (neutralized in favor of pre-computed Số lượng quy đổi)",
            patched,
        )
        patched = patched.replace("['Số lượng cơ bản']", "['Số lượng quy đổi']").replace('["Số lượng cơ bản"]', '["Số lượng quy đổi"]')
        patched = patched.replace("['sl_co_ban']", "['Số lượng quy đổi']").replace('["sl_co_ban"]', '["Số lượng quy đổi"]')
        patched = patched.replace("['Quantity_in_Base_Unit']", "['Số lượng quy đổi']").replace('["Quantity_in_Base_Unit"]', '["Số lượng quy đổi"]')
        patched = patched.replace("['Số lượng lẻ']", "['Số lượng quy đổi']").replace('["Số lượng lẻ"]', '["Số lượng quy đổi"]')

    if is_product_ranking:
        for rev_col in ("Thành tiền", "ThanhTien", "Amount", "Tổng tiền", "TongTien", "tien_hang", "Tiền hàng"):
            patched = re.sub(
                rf"\.groupby\(([^)]*)\)\[\s*['\"]{rev_col}['\"]\s*\]\.sum\(\)",
                r".groupby(\1)['Số lượng quy đổi'].sum()",
                patched
            )
        for raw_q in ("Số lượng", "Số lượng sản phẩm", "so_luong", "SL", "Số lượng hàng", "SoLuong"):
            patched = patched.replace(f"['{raw_q}']", "['Số lượng quy đổi']").replace(
                f'["{raw_q}"]', '["Số lượng quy đổi"]'
            )

    # Weighted Average for import price (Rulebook §5 line 65)
    if any(k in qtext for k in ("gia nhap binh quan", "nhap vao gia binh quan", "gia nhap trung binh", "don gia nhap")) and any(k in qtext for k in ("binh quan", "trung binh")):
        for tname, df in exec_tables.items():
            price_col = next((c for c in df.columns if compact_identifier(c) in ("dongia", "dongianhap", "gianhap")), None)
            qty_col = next((c for c in df.columns if compact_identifier(c) in ("soluong", "sl", "slnhap", "soluongnhap")), None)
            amt_col = next((c for c in df.columns if compact_identifier(c) in ("thanhtien", "cantrancc", "tonggiatrinhap")), None)
            if price_col and qty_col:
                if amt_col:
                    patched = re.sub(
                        rf"(\b{tname}(?:\[.+?\])?)\[\s*['\"]{price_col}['\"]\s*\]\.mean\(\)",
                        rf"(\1['{amt_col}'].sum() / \1['{qty_col}'].sum())",
                        patched,
                    )
                else:
                    patched = re.sub(
                        rf"(\b{tname}(?:\[.+?\])?)\[\s*['\"]{price_col}['\"]\s*\]\.mean\(\)",
                        rf"((\1['{qty_col}'] * \1['{price_col}']).sum() / \1['{qty_col}'].sum())",
                        patched,
                    )

    # Rulebook §3 lines 43–44 & §5 line 71: Redirect raw quantity ('Số lượng') to base-unit quantity ('Số lượng quy đổi')
    # when ranking products, comparing price-change quantities, or summing base-unit quantities.
    has_sl_quy_doi = any("Số lượng quy đổi" in df.columns for df in exec_tables.values())
    if has_sl_quy_doi:
        large_units: Set[str] = set()
        for tname, df in exec_tables.items():
            for col in df.columns:
                if compact_identifier(col) in ("donviquydoi", "dvnhap", "dvlon", "donvilon", "dvtquydoi"):
                    for u in df[col].dropna().astype(str).unique():
                        u_norm = normalize_text(u)
                        if u_norm:
                            large_units.add(u_norm)
        asks_large_unit = any(
            bool(re.search(rf"\b(?:bao nhieu|may|so|tinh theo)\s+{re.escape(lu)}\b", qtext))
            for lu in large_units
        )
        asks_base_qty = (is_product_ranking or any(
            k in qtext
            for k in (
                "ban chay", "chay nhat", "ban nhieu nhat", "hot nhat",
                "tang gia", "doi gia", "len gia",
                "ban duoc", "ban ra", "top", "mon nao", "san pham nao", "hang nao",
            )
        )) and not question_is_pure_revenue(question) and not is_order_counting
        if asks_base_qty and not asks_large_unit:
            if exec_tables is tables:
                exec_tables = {k: v.copy() for k, v in tables.items()}
            for df in exec_tables.values():
                if "Số lượng quy đổi" not in df.columns:
                    for raw_q in ("Số lượng", "Số lượng sản phẩm", "so_luong", "SL"):
                        if raw_q in df.columns:
                            df["Số lượng quy đổi"] = df[raw_q]
                            break
            for raw_q in ("Số lượng", "Số lượng sản phẩm", "so_luong"):
                patched = patched.replace(f"['{raw_q}']", "['Số lượng quy đổi']").replace(
                    f'["{raw_q}"]', '["Số lượng quy đổi"]'
                )

    # Rulebook §6 lines 89-91: Pharmacy active ingredient exact match (single active ingredient vs compound combinations)
    if any(k in qtext for k in ("hoat chat", "hoạt chất", "thanh phan chinh", "thành phần chính")):
        ing_pattern = r"([A-Za-z0-9_]+\[\s*['\"](?:Hoạt chất|hoat_chat|HoatChat|hoatchat|TenHoatChat|ThanhPhanChinh|thanh_phan_chinh|ThanhPhan|thanh_phan|HoatChatChinh)['\"]\s*\])\.str\.contains\(\s*r?['\"]([^'\"]+)['\"][^)]*\)"
        patched = re.sub(
            ing_pattern,
            lambda m: f"({m.group(1)}.astype(str).str.strip().str.lower() == '{m.group(2).strip().lower()}')",
            patched,
            flags=re.IGNORECASE,
        )

    # Rulebook §6 lines 95-97: Delivery trips deduplication (distinct trips by Order ID + Trip / Lượt giao)
    if any(k in qtext for k in ("dot giao", "luot giao", "xe da giao", "xe cho", "chuyen giao", "dot hang")):
        for tname, df in exec_tables.items():
            trip_col = next((c for c in df.columns if compact_identifier(c) in ("luotgiao", "dotgiao", "chuyengiao", "langiao")), None)
            order_col = next((c for c in df.columns if compact_identifier(c) in ("maphieu", "madon", "madonhang", "soct", "sophieu")), None)
            if trip_col and order_col:
                patched = re.sub(
                    rf"(\b{tname}(?:\[.+?\])?)\[\s*['\"]{trip_col}['\"]\s*\]\.(?:nunique|count)\(\)",
                    rf"\1.drop_duplicates(subset=['{order_col}', '{trip_col}']).shape[0]",
                    patched,
                )
                patched = re.sub(
                    rf"len\(\s*(\b{tname}(?:\[.+?\])?)\s*\)",
                    rf"\1.drop_duplicates(subset=['{order_col}', '{trip_col}']).shape[0]",
                    patched,
                )

    if question_is_pure_revenue(question):
        # 1. First priority: if tables have 'Tiền hàng trước giảm' (pre-discount gross revenue)
        has_tien_hang_truoc_giam = any(
            "Tiền hàng trước giảm" in df.columns for df in exec_tables.values()
        )
        if has_tien_hang_truoc_giam:
            for bad in ("Tong tien", "Tong_tien", "Tổng tiền", "TongTien", "Tổng sau giảm giá", "Tổng bill", "Thanh toán"):
                patched = patched.replace(f"['{bad}']", "['Tiền hàng trước giảm']").replace(f'["{bad}"]', '["Tiền hàng trước giảm"]')
        else:
            # 2. Injected or existing 'Tổng tiền hàng'
            has_injected_tong_tien_hang = any("Tổng tiền hàng" in df.columns for df in exec_tables.values())
            if has_injected_tong_tien_hang:
                for old_col in ("Tổng bill", "Tong tien", "Tong_tien", "Tổng đơn", "Thanh toán", "Tổng sau giảm giá", "Phải thu"):
                    patched = patched.replace(f"['{old_col}']", "['Tổng tiền hàng']").replace(f'["{old_col}"]', '["Tổng tiền hàng"]')
            # 3. 'Tien hang'
            has_tien_hang = any("Tien hang" in df.columns for df in exec_tables.values())
            if has_tien_hang:
                for bad in ("TongTien", "Tong tien", "Tong_tien", "Tổng tiền", "Tổng sau giảm giá"):
                    patched = patched.replace(f"['{bad}']", "['Tien hang']").replace(f'["{bad}"]', '["Tien hang"]')

    return patched, exec_tables


def format_accounting_context(
    tables: Dict[str, pd.DataFrame],
    question: str = "",
) -> Optional[str]:
    """
    Build a concise schema note for the prompt highlighting detected status
    columns and Rulebook §1 revenue columns.
    """
    notes: List[str] = []
    seen_bases: Set[str] = set()
    for name, df in tables.items():
        base = name.split("__")[0]
        if not name.endswith("__monthly_combined") and f"{base}__monthly_combined" in tables:
            continue
        if base in seen_bases:
            continue
        seen_bases.add(base)

        det = detect_status_column(df)
        if det:
            scol, mp = det
            cancel_vals = [v for v, c in mp.items() if c == "cancel"]
            return_vals = [v for v, c in mp.items() if c == "return"]
            valid_vals = [v for v, c in mp.items() if c in ("completed", "shipping")]
            status_desc = f"valid={valid_vals}"
            if cancel_vals:
                status_desc += f", cancelled_only={cancel_vals}"
            if return_vals:
                status_desc += f", returned_only={return_vals}"
            notes.append(
                f"- Table `{name}` status column `{scol}`: {status_desc}."
            )
        if "Tiền hàng trước giảm" in df.columns and "Tong tien" in df.columns:
            notes.append(
                f"- Table `{name}`: Use `'Tiền hàng trước giảm'` for gross revenue (Rulebook §1); do NOT use `'Tong tien'` or `'Đã thu'`."
            )
        elif "Tổng tiền hàng" in df.columns and "Tổng sau giảm giá" in df.columns:
            notes.append(
                f"- Table `{name}`: Use `'Tổng tiền hàng'` for revenue (Rulebook §1); do NOT use `'Tổng sau giảm giá'` or `'Khách đã trả'`."
            )
        elif "Tien hang" in df.columns and "TongTien" in df.columns:
            notes.append(
                f"- Table `{name}`: Use `'Tien hang'` for revenue (Rulebook §1); do NOT use `'TongTien'`."
            )
        elif "Phải thu" in df.columns and "Thanh toán" in df.columns:
            notes.append(
                f"- Table `{name}`: In `don_hang`, `'Phải thu'` and `'Thanh toán'` are POST-discount (net paid). Pre-discount gross revenue (Rulebook §1) is the sum of `'Tiền bán'` across item lines in `chi_tiet_don`."
            )

    # Multi-line order tables and deduplication
    has_multi_line_orders = any(
        find_order_id_column(df) is not None and df[find_order_id_column(df)].duplicated().any()
        for df in tables.values()
    )
    if has_multi_line_orders:
        notes.append(
            "- Order Counting & Rate: If an order table has multiple lines per order, always count unique orders (`df['Mã đơn'].nunique()`), NOT row counts (`len(df)` or `df.shape[0]`)."
        )
        notes.append(
            "- Average Order Value (AOV): For tables with multiple lines per order, aggregate items per order first: `df.groupby('Mã đơn')['Thành tiền'].sum().mean()`."
        )

    # General Rulebook §3 note for product ranking: service fee exclusion
    notes.append(
        "- Product Ranking (Rulebook §3): Dòng 'Phí phục vụ' không phải mặt hàng: không xếp hạng, không đếm (nhưng vẫn tính vào doanh thu)."
    )

    # Relational merge fan-out note
    notes.append(
        "- Merge Fan-out: When joining order header with order detail tables, do NOT sum order-level totals (like 'Tổng tiền hàng') on the merged table as it duplicates revenue; sum detail-level 'Thành tiền' instead."
    )

    # General Rulebook §10 note for base unit conversion
    has_quydoi = any("quydoi" in k.lower() or "don_vi" in k.lower() for k in tables)
    if has_quydoi:
        notes.append(
            "- Unit Conversion (Rulebook §3 & §10): Khi xếp hạng bán chạy hoặc tính tổng số lượng, phải quy đổi về đơn vị cơ bản bằng bảng quy đổi (`DanhSachQuyDoi` / `don_vi_sp`)."
        )

    # Rulebook §5 Unsold goods guidance
    qtext = normalize_text(question)
    if any(k in qtext for k in ("khong ban duoc", "ko ban duoc", "ko ban dc", "hang e", "mat hang e", "mat hang bi e", "bi e", "ban ko dc", "e hang")):
        notes.append(
            "- Hàng ế trong kỳ (Rulebook §5 line 69): Là các mặt hàng ĐÃ TỪNG BÁN trước kỳ (có trong đơn hàng trước ngày bắt đầu kỳ) nhưng trong kỳ có số lượng bán = 0: `len(set(df[df[date_col] < start_date][prod_col]) - set(df[(df[date_col] >= start_date) & (df[date_col] < (end_date + pd.Timedelta(days=1)))][prod_col]))`. KHÔNG lấy từ danh mục hàng."
        )

    # Rulebook §5 Weighted average import price guidance
    if any(k in qtext for k in ("gia nhap binh quan", "nhap vao gia binh quan", "gia nhap trung binh", "don gia nhap")):
        notes.append(
            "- Giá nhập bình quân một đơn vị (Rulebook §5 line 65): Phải tính bình quân gia quyền theo số lượng nhập của đơn vị đó: `(df['Số lượng'] * df['Đơn giá']).sum() / df['Số lượng'].sum()`. Chú ý: dùng cột 'Số lượng' (theo đơn vị được hỏi), KHÔNG chia cho 'Số lượng (ĐVT cơ bản)'. Làm tròn về đồng."
        )

    # Rulebook §4 line 58 Debt added & Debt paid guidance
    if any(k in qtext for k in ("no them", "ghi no them", "mua chiu them", "tra no", "tra bot no", "thu no", "thu hoi no", "thu hoi cong no", "tra duoc", "mua chiu")):
        notes.append(
            "- Công nợ phát sinh và trả nợ trong kỳ (Rulebook §4 line 58):"
            "\n  + Nợ thêm trong kỳ: Là phần chưa thanh toán lúc mua của các đơn ghi nợ/mua chịu phát sinh trong kỳ: `(Tổng đơn - Tiền trả lúc mua)` (hoặc cột `Còn nợ` của đơn nợ lúc lập). Không tính đơn đã hủy/trả hàng."
            "\n  + Trả nợ / Thu nợ trong kỳ: Là tổng số tiền thu/trả nợ phát sinh trong kỳ trên Sổ quỹ / Phiếu thu / Thanh toán (chỉ tính tiền thu nợ cho các đơn nợ, không tính tiền khách trả ngay lúc mua)."
            "\n  + Nếu câu hỏi yêu cầu ghi 2 số theo thứ tự: trả về danh sách 2 số `[nợ_thêm, trả_được]`."
        )

    # Rulebook §6 Blister/Pill even/odd inventory guidance
    if any(k in qtext for k in ("vi chan", "le vien", "chan bao nhieu vi", "le bao nhieu vien")):
        notes.append(
            "- Tồn kho vỉ chẵn, viên lẻ (Rulebook §6): Tìm số viên trên mỗi vỉ trong quy cách đóng gói ở tên thuốc (ví dụ: 'hộp 10 vỉ x 8 viên' -> 8 viên/vỉ, '10 vỉ x 10 viên' -> 10 viên/vỉ). Số vỉ chẵn = tồn_kho // viên_mỗi_vỉ; số viên lẻ = tồn_kho % viên_mỗi_vỉ. Trả về danh sách 2 số: `[vỉ_chẵn, viên_lẻ]`."
        )

    # Rulebook §5 line 71 Price change comparison guidance
    if any(k in qtext for k in ("tang gia", "doi gia", "len gia", "giam gia")) and any(k in qtext for k in ("ngay ke tu", "ngay truoc", "ngay lien truoc", "ngay ngay truoc")):
        notes.append(
            "- So sánh lượng bán trước/sau ngày đổi giá D trong N ngày (Rulebook §5 line 71): Khoảng sau là `[D, D + pd.Timedelta(days=N)]`; khoảng ngay trước là `[D - pd.Timedelta(days=N), D]`. % thay đổi = `(lượng_sau - lượng_trước) / lượng_trước * 100`. Tính theo đơn vị câu hỏi yêu cầu (viên, ly, con, lon...)."
        )

    # Rulebook §6 Active ingredient guidance
    if any(k in qtext for k in ("hoat chat", "khe don", "thuoc ke don", "ke don")):
        notes.append(
            "- Hoạt chất & Thuốc kê đơn (Rulebook §6): Doanh thu theo hoạt chất là thuốc có hoạt chất khớp đúng như vậy trong danh mục thuốc. Thuốc phối hợp nhiều hoạt chất là một loại riêng, không tính vào từng hoạt chất riêng lẻ. Tỉ trọng (%): lấy doanh thu thuốc kê đơn chia cho tổng doanh thu toàn bộ (cùng loại doanh thu trước giảm và loại trừ đơn hủy/trả)."
        )

    if not notes:
        return None
    return "ACCOUNTING & RULEBOOK COLUMN NOTES:\n" + "\n".join(notes)
