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
    r"\b(huy|tra hang|tra lai|hoan|no|cong no|gia ban|dot giao|ti le|ty le)\b"
)
_RANK_CUSTOMER_RE = re.compile(
    r"\b(khach hang|khach nao|ai mua|khach mua)\b.*\b(nhieu nhat|cao nhat|top|xep hang|nhieu don|nhieu tien|nhiu tien)\b"
    r"|\b(top|nhieu nhat|cao nhat)\b.*\b(khach)\b"
)
_AOV_RE = re.compile(
    r"\b(trung binh moi don|binh quan moi don|binh quan moi bill|gia tri trung binh mot don"
    r"|doanh thu trung binh.*don|dthu trung binh.*don|doanh thu binh quan.*don|doanh thu binh quan.*bill)\b"
)
_PURE_REVENUE_RE = re.compile(
    r"\b(doanh thu|dthu|revenue|ban duoc bao nhieu tien|ban dc bao nhieu tien"
    r"|ban duoc bn tien|ban dc nhieu tien|mang ve bao nhieu doanh thu|mang ve bn dthu"
    r"|mang ve bnhiu dthu|doanh thu binh quan)\b"
)
_DEBT_OR_PAID_RE = re.compile(
    r"\b(no|cong no|da tra|da thu|thuc thu|con no|phai thu)\b"
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


def question_needs_all_statuses(question: str) -> bool:
    """True when the question asks about cancelled/returned orders, rates, debt, selling price, or delivery batches."""
    return bool(_NEEDS_ALL_STATUS_RE.search(normalize_text(question)))


def question_is_customer_ranking(question: str) -> bool:
    """True when the question ranks or finds top individual customers."""
    return bool(_RANK_CUSTOMER_RE.search(normalize_text(question)))


def question_is_average_order_value(question: str) -> bool:
    """True when the question asks for average value/revenue per order (Rulebook §3 line 48)."""
    return bool(_AOV_RE.search(normalize_text(question)))


def question_is_pure_revenue(question: str) -> bool:
    """True when the question asks for revenue and not debt/paid amounts."""
    q = normalize_text(question)
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
                if any(
                    t in c_comp for t in ("tenkhach", "khachhang", "benmua", "tenkh")
                ) and not any(x in c_comp for x in ("nhom", "loai", "ma")):
                    s_col = work[col].dropna().astype(str)
                    vc_map = {
                        v: normalize_text(v) in _WALKIN_VALUES for v in s_col.unique()
                    }
                    walkin_mask = work[col].map(vc_map).fillna(False)
                    if walkin_mask.any():
                        work = work.loc[~walkin_mask].copy()

        out[name] = work

    # Propagate excluded order IDs to detail tables sharing an order ID column
    if filter_status and bad_order_ids:
        for name, work in list(out.items()):
            oid = find_order_id_column(work)
            if oid is not None:
                drop = work[oid].isin(bad_order_ids)
                if drop.any():
                    out[name] = work.loc[~drop].copy()

    return out


_REDUNDANT_STATUS_COLS_PATTERN = (
    r"(?:Trạng thái|Trạng thái đơn hàng|Trạng thái xử lý|Trạng thái đơn|"
    r"Tình trạng|Tình trạng đơn|Kết quả|Xử lý|Status|Ghi chú TT|trang_thai|"
    r"TinhTrang|Unnamed: [45])"
)


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

    qtext = normalize_text(question)
    needs_all = question_needs_all_statuses(question)
    patched = code
    exec_tables = tables

    if not needs_all:
        patched = re.sub(
            rf"([A-Za-z0-9_]+\[\s*['\"]{_REDUNDANT_STATUS_COLS_PATTERN}['\"]\s*\])\s*"
            r"==\s*['\"](?:Hoàn thành|Đã hoàn thành|completed|xong|đã xong|Đã giao|Thành công|đã lấy|ok)['\"]",
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

        # If question asks to count orders or calculate order cancellation rate (Rulebook §1),
        # ensure .shape[0] on cancelled rows deduplicates by order ID for multi-line order tables.
        is_order_counting = any(
            k in qtext for k in ("dem so don", "may don", "bao nhieu don", "ti le don", "ti le huy", "% don", "so don")
        )
        if is_order_counting:
            for tname, df in exec_tables.items():
                oid = find_order_id_column(df)
                if oid is not None and df[oid].duplicated().any():
                    # Rewrite .shape[0] on order-level calculations
                    patched = re.sub(
                        rf"(\[[^\]]*__CANCEL__[^\]]*\])\.shape\[0\]",
                        rf"\1.drop_duplicates(subset=['{oid}']).shape[0]",
                        patched,
                    )

    if question_is_pure_revenue(question):
        has_tong_tien_hang = any(
            "Tổng tiền hàng" in df.columns and "Tổng sau giảm giá" in df.columns
            for df in tables.values()
        )
        if has_tong_tien_hang:
            patched = patched.replace("['Tổng sau giảm giá']", "['Tổng tiền hàng']")
            patched = patched.replace('["Tổng sau giảm giá"]', '["Tổng tiền hàng"]')
        has_tien_hang = any(
            "Tien hang" in df.columns and "TongTien" in df.columns
            for df in tables.values()
        )
        if has_tien_hang:
            patched = patched.replace("['TongTien']", "['Tien hang']")
            patched = patched.replace('["TongTien"]', '["Tien hang"]')

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
        if "Tổng tiền hàng" in df.columns and "Tổng sau giảm giá" in df.columns:
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

    # General Rulebook §3 note for product ranking: service fee exclusion
    notes.append(
        "- Product Ranking (Rulebook §3): Dòng 'Phí phục vụ' không phải mặt hàng: không xếp hạng, không đếm (nhưng vẫn tính vào doanh thu)."
    )

    # General Rulebook §10 note for base unit conversion
    has_quydoi = any("quydoi" in k.lower() or "don_vi" in k.lower() for k in tables)
    if has_quydoi:
        notes.append(
            "- Unit Conversion (Rulebook §3 & §10): Khi xếp hạng bán chạy hoặc tính tổng số lượng, phải quy đổi về đơn vị cơ bản bằng bảng quy đổi (`DanhSachQuyDoi` / `don_vi_sp`)."
        )
    if not notes:
        return None
    return "ACCOUNTING & RULEBOOK COLUMN NOTES:\n" + "\n".join(notes)
