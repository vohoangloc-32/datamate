"""
data_loader.py — Universal Data Loader for DataMate

Handles CSV, XLSX (multi-sheet), and SQLite files.
Produces pandas DataFrames with auto-detected types, normalized column names,
and encoding detection for messy real-world Vietnamese data.
"""

import os
import re
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import unicodedata

from .accounting_cleaner import (
    dedup_order_table_rows,
    normalize_text,
    remove_subheader_and_subtotal_rows,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CURRENCY_UNIT_TRAILING_RE = re.compile(r"(?i)\s*(?:đ|vnd|dong|d)$")
_CURRENCY_COMBO_RE = re.compile(r"(\d+)\s*(?:tr|trieu)\s*(\d{1,3})", re.IGNORECASE)
_CURRENCY_MILLIONS_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:tr|trieu)", re.IGNORECASE)
_CURRENCY_THOUSANDS_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*k", re.IGNORECASE)
_CURRENCY_DOT_THOUSANDS_RE = re.compile(r"\d{1,3}(?:\.\d{3})+")
_CURRENCY_COMMA_THOUSANDS_RE = re.compile(r"\d{1,3}(?:,\d{3})+")


def parse_vietnamese_currency_str(val: object) -> object:
    """
    Parse informal Vietnamese currency strings per Rulebook §8 lines 113–122.
    Handles:
      - '25.000đ', '-53.500đ', '575.000 đ', '50.000 VND'
      - '1tr440', '1tr140', '9tr960' (millions + thousands combo)
      - '16,56tr', '1,84tr' (decimal millions)
      - '86k', '1168k', '29200k' (thousands)
      - '1.016.000', '1,250,000' (standard thousand-separated numbers)
    Returns int/float if successfully parsed, else original val.
    """
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s or s.lower() == "nan":
        return val

    sign = -1 if s.startswith("-") else 1
    s_clean = s.lstrip("+-").strip()
    s_no_unit = _CURRENCY_UNIT_TRAILING_RE.sub("", s_clean).strip()

    # 1. Combo millions + thousands: e.g. 1tr440 -> 1_440_000
    m_combo = _CURRENCY_COMBO_RE.fullmatch(s_no_unit)
    if m_combo:
        millions = int(m_combo.group(1))
        rem = m_combo.group(2)
        thousands = int(rem.ljust(3, "0")) * 1000
        return sign * (millions * 1_000_000 + thousands)

    # 2. Millions with tr / trieu: e.g. 16,56tr -> 16_560_000
    m_tr = _CURRENCY_MILLIONS_RE.fullmatch(s_no_unit)
    if m_tr:
        num_str = m_tr.group(1).replace(",", ".")
        return sign * round(float(num_str) * 1_000_000)

    # 3. Thousands with k: e.g. 86k -> 86_000, 1168k -> 1_168_000
    m_k = _CURRENCY_THOUSANDS_RE.fullmatch(s_no_unit)
    if m_k:
        num_str = m_k.group(1).replace(",", ".")
        return sign * round(float(num_str) * 1_000)

    # 4. Standard dot-separated thousands: e.g. 25.000 -> 25000
    if _CURRENCY_DOT_THOUSANDS_RE.fullmatch(s_no_unit):
        return sign * int(s_no_unit.replace(".", ""))

    # 5. Standard comma-separated thousands: e.g. 1,250,000 -> 1250000
    if _CURRENCY_COMMA_THOUSANDS_RE.fullmatch(s_no_unit):
        return sign * int(s_no_unit.replace(",", ""))

    # 6. Plain float / int
    try:
        f = float(s_no_unit.replace(",", ""))
        return sign * (int(f) if f.is_integer() else f)
    except Exception:
        pass

    return val


def repair_mojibake_name(name: str) -> str:
    """Repair CP437/CP1252 mojibake from zip extractions (e.g., 'Danh s├ích h├áng h├│a' -> 'Danh sách hàng hóa')."""
    if not name:
        return name
    if any(c in name for c in "├─│┤┬┴┼ß"):
        try:
            return name.encode("cp437").decode("utf-8")
        except Exception:
            pass
    return name


def _normalize_col_name(name: str) -> str:
    """Normalize a column name: Unicode NFC, strip, collapse whitespace."""
    name = repair_mojibake_name(str(name))
    name = unicodedata.normalize("NFC", name)
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    return name


def _compact_col_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(name).casefold())
    normalized = "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    ).replace("đ", "d")
    return re.sub(r"[^a-z0-9]", "", normalized)


def _is_date_column(name: str) -> bool:
    """Recognize common English and Vietnamese date/time column names."""
    compact = _compact_col_name(name)
    return any(token in compact for token in ("date", "datetime", "time", "ngay", "thoigian"))


def _monthly_sheet_period(name: str) -> Optional[str]:
    """Return YYYY-MM for sheet names that clearly identify a month."""
    compact = _compact_col_name(name)
    match = re.fullmatch(r"(?:thang|t)?(0?[1-9]|1[0-2])(20\d{2})", compact)
    if not match:
        return None
    month, year = match.groups()
    return f"{year}-{int(month):02d}"


def _is_order_id_column(name: str) -> bool:
    compact = _compact_col_name(name)
    return any(
        token in compact
        for token in (
            "madon", "mahd", "sohd", "mahoadon", "sohoadon",
            "orderid", "orderno", "ordernumber", "machungtu",
            "sochungtu", "soct", "madh", "sodon", "mabill",
            "sophieu", "maphieu", "magd", "manhaphang",
            "manhap", "madonnhap", "phieunhap", "donnhap",
        )
    )


def _is_order_metadata_column(name: str) -> bool:
    compact = _compact_col_name(name)
    return (
        _is_order_id_column(name)
        or any(token in compact for token in (
            "ngaydat",
            "orderdate",
            "trangthaidonhang",
            "trangthai",
            "tinhtrang",
            "orderstatus",
            "tenkhachhang",
            "customername",
        ))
    )


def _fill_merged_order_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Fill merged order-level cells across product rows in order worksheets."""
    order_id_columns = [
        column for column in df.columns if _is_order_id_column(column)
    ]
    if not order_id_columns:
        return df

    order_id_column = order_id_columns[0]
    order_keys = df[order_id_column].ffill()
    if not df[order_id_column].isna().any():
        return df

    metadata_columns = [
        column for column in df.columns
        if _is_order_metadata_column(column)
    ]
    for column in metadata_columns:
        if column == order_id_column:
            df[column] = order_keys
        else:
            df[column] = df[column].groupby(order_keys, sort=False).ffill()
    return df


def _coerce_date_column(values: pd.Series, default_year: Optional[int] = None) -> Optional[pd.Series]:
    """Parse date-like columns, including mixed Excel serial-day and text dates."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values

    non_null = values.dropna()
    if non_null.empty:
        return None

    # If all non-null values are numeric outside Excel serial range (e.g. integer years 2024, 2025), keep as numeric
    numeric = pd.to_numeric(values, errors="coerce")
    valid_numeric = numeric.dropna()
    if len(valid_numeric) == len(non_null) and not valid_numeric.between(20000, 80000).any():
        return None

    result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns]")

    # 1. Excel serial day values (20000 to 80000, representing 1954 to 2119)
    excel_mask = numeric.between(20000, 80000)
    if excel_mask.any():
        parsed_excel = pd.to_datetime(
            numeric[excel_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )
        result.loc[excel_mask] = parsed_excel

    # 2. Text dates for elements not parsed as Excel serial
    remaining_mask = values.notna() & ~excel_mask
    if remaining_mask.any():
        date_text = values[remaining_mask].astype("string").str.strip()
        # Exclude bare 4-digit years like "2024"
        is_bare_year = date_text.str.match(r"^\d{4}$", na=False)
        date_text_to_parse = date_text[~is_bare_year]
        if not date_text_to_parse.empty:
            # Rulebook §8, line 113: Supply sheet year or modal year for bare DD/MM dates
            target_year = default_year
            if target_year is None:
                yr_extract = values.astype(str).str.extract(r"\b(20\d{2})\b")[0].dropna()
                if not yr_extract.empty:
                    target_year = int(yr_extract.mode()[0])
                elif excel_mask.any() and not result.dropna().empty:
                    target_year = int(result.dropna().dt.year.mode()[0])
            if target_year is not None:
                date_text_to_parse = date_text_to_parse.str.replace(
                    r"^\s*(\d{1,2})[/-](\d{1,2})\s*$",
                    rf"\1/\2/{target_year}",
                    regex=True,
                )

            date_text_to_parse = date_text_to_parse.str.replace(
                r"(?i)\bth(?:áng|g)\.?\s*",
                "/",
                regex=True,
            )
            date_text_to_parse = date_text_to_parse.str.replace(r"\s+", "/", regex=True)
            date_text_to_parse = date_text_to_parse.str.replace(r"/+", "/", regex=True)

            parsed_text = pd.to_datetime(
                date_text_to_parse,
                errors="coerce",
                dayfirst=True,
                format="mixed",
            )
            iso_dates = date_text_to_parse.str.match(r"^\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}(?:\b|T)", na=False)
            if iso_dates.any():
                parsed_iso = pd.to_datetime(
                    date_text_to_parse.loc[iso_dates],
                    errors="coerce",
                    yearfirst=True,
                    format="mixed",
                )
                parsed_text.loc[iso_dates] = parsed_iso

            # Clamp to plausible calendar years to avoid OutOfBoundsDatetime (e.g. year 1 AD)
            valid_year_mask = parsed_text.dt.year.between(1950, 2100)
            parsed_text.loc[~valid_year_mask] = pd.NaT
            result.loc[date_text_to_parse.index] = parsed_text

    valid_dates = result.dropna()
    if len(valid_dates) >= 0.5 * len(non_null):
        years = valid_dates.dt.year
        if (years >= 2000).mean() >= 0.75 and (years <= 2050).mean() >= 0.75:
            return result
    return None


def _detect_encoding(filepath: str) -> str:
    """Detect file encoding using charset-normalizer (faster & more accurate
    than chardet for Vietnamese text)."""
    try:
        from charset_normalizer import from_path
        result = from_path(filepath)
        best = result.best()
        if best is not None:
            enc = str(best.encoding)
            logger.debug("Detected encoding %s for %s", enc, filepath)
            return enc
    except Exception:
        pass
    return "utf-8"


def _coerce_types(df: pd.DataFrame, default_year: Optional[int] = None) -> pd.DataFrame:
    """Try to coerce object/string columns to numeric or datetime where appropriate."""
    for col in df.columns:
        if _is_date_column(col):
            parsed_dates = _coerce_date_column(df[col], default_year=default_year)
            if parsed_dates is not None:
                df[col] = parsed_dates
                continue

        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue

            # 1. Try direct numeric
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().sum() > 0.6 * len(non_null):
                df[col] = numeric
                continue

            # 2. Try numeric with thousand-separator commas removed (e.g., "1,250,000" -> 1250000)
            cleaned_num = non_null.astype(str).str.replace(",", "").str.strip()
            numeric_cleaned = pd.to_numeric(cleaned_num, errors="coerce")
            if numeric_cleaned.notna().sum() > 0.7 * len(non_null):
                full_cleaned = df[col].astype(str).str.replace(",", "").str.strip()
                df[col] = pd.to_numeric(full_cleaned, errors="coerce")
                continue

            # 2b. Try Vietnamese currency parsing (e.g., '25.000đ', '86k', '1tr440', '16,56tr')
            sample_curr = non_null.head(20).astype(str)
            if any(
                _CURRENCY_UNIT_TRAILING_RE.search(v)
                or _CURRENCY_COMBO_RE.search(v)
                or _CURRENCY_MILLIONS_RE.search(v)
                or _CURRENCY_THOUSANDS_RE.search(v)
                for v in sample_curr
            ):
                parsed_curr = non_null.map(parse_vietnamese_currency_str)
                numeric_curr = pd.to_numeric(parsed_curr, errors="coerce")
                if numeric_curr.notna().sum() > 0.7 * len(non_null):
                    full_parsed = df[col].map(parse_vietnamese_currency_str)
                    df[col] = pd.to_numeric(full_parsed, errors="coerce")
                    continue

            # 2c. Try Quantity with unit string parsing (e.g. '1 chai', '2 ly', '10 hop')
            compact_c = _compact_col_name(col)
            if any(t in compact_c for t in ("soluong", "sl", "qty", "quantity")) and not any(t in compact_c for t in ("ma", "id", "ten")):
                _qty_re = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)\s+[a-zA-ZÀ-ỹ]+\s*$")
                sample_qty = non_null.head(30).astype(str)
                if any(_qty_re.match(v) for v in sample_qty):
                    def _extract_qty(val: object) -> object:
                        if isinstance(val, (int, float)):
                            return val
                        s_v = str(val).strip()
                        m_q = _qty_re.match(s_v)
                        if m_q:
                            num_str = m_q.group(1).replace(",", ".")
                            try:
                                return float(num_str) if "." in num_str else int(num_str)
                            except Exception:
                                return val
                        return val

                    extracted_sample = non_null.map(_extract_qty)
                    numeric_qty = pd.to_numeric(extracted_sample, errors="coerce")
                    if numeric_qty.notna().sum() > 0.7 * len(non_null):
                        full_extracted = df[col].map(_extract_qty)
                        df[col] = pd.to_numeric(full_extracted, errors="coerce")
                        continue

            # 3. Try datetime only if sample strings contain date separators and years are plausible
            sample_str = " ".join(non_null.head(5).astype(str))
            if any(sep in sample_str for sep in ("/", "-")):
                try:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", UserWarning)
                        dt = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
                    valid_dt = dt.dropna()
                    if len(valid_dt) > 0.6 * len(non_null):
                        years = valid_dt.dt.year
                        if (years >= 2000).mean() > 0.8 and (years <= 2050).mean() > 0.8:
                            df[col] = dt
                            continue
                except Exception:
                    pass

            # 4. If column remains string/object: clean whitespace and harmonize casing
            mask = df[col].notna() & df[col].map(lambda x: isinstance(x, str))
            if mask.any():
                df.loc[mask, col] = (
                    df.loc[mask, col]
                    .astype(str)
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                )
                s_non_null = df.loc[mask, col]
                if 0 < s_non_null.str.casefold().nunique() < s_non_null.nunique() <= 1000:
                    mode_map = {}
                    for lower_val, grp in s_non_null.groupby(s_non_null.str.casefold()):
                        mode_map[lower_val] = grp.value_counts().index[0]
                    df.loc[mask, col] = s_non_null.str.casefold().map(mode_map)
    return df


# ---------------------------------------------------------------------------
# Loaders per format
# ---------------------------------------------------------------------------

def _load_csv(filepath: str) -> Dict[str, pd.DataFrame]:
    """Load a CSV file, returning {filename_stem: DataFrame}."""
    raw_stem = Path(filepath).stem
    stem = repair_mojibake_name(raw_stem)
    encoding = _detect_encoding(filepath)

    # Try common separators
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(filepath, encoding=encoding, sep=sep,
                             on_bad_lines="warn", engine="python")
            if len(df.columns) > 1:
                break
        except Exception:
            continue
    else:
        df = pd.read_csv(filepath, encoding=encoding, on_bad_lines="warn")

    df.columns = [_normalize_col_name(c) for c in df.columns]
    df = remove_subheader_and_subtotal_rows(df)
    df = _fill_merged_order_metadata(df)
    df = _coerce_types(df)
    df = dedup_order_table_rows(df)
    logger.info("Loaded CSV %s: %d rows × %d cols", filepath, len(df), len(df.columns))
    res = {stem: df}
    if raw_stem != stem:
        res[raw_stem] = df
    return res


def _load_xlsx(filepath: str) -> Dict[str, pd.DataFrame]:
    """Load an Excel file (all sheets), returning {sheet_name: DataFrame}."""
    result = {}
    monthly_sheets = []
    raw_stem = Path(filepath).stem
    stem = repair_mojibake_name(raw_stem)
    xls = pd.ExcelFile(filepath, engine="openpyxl")
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
            unnamed_columns = sum(
                str(column).startswith("Unnamed:") for column in df.columns
            )
            first_row_non_null = (
                int(df.iloc[0].notna().sum()) if not df.empty else 0
            )
            if (
                len(df.columns) > 1
                and unnamed_columns >= len(df.columns) / 2
                and first_row_non_null <= max(2, len(df.columns) // 4)
            ):
                raw_df = pd.read_excel(xls, sheet_name=sheet, header=None)
                header_row = _find_xlsx_header_row(raw_df)
                if header_row > 0:
                    headers = raw_df.iloc[header_row].tolist()
                    columns = [
                        _normalize_col_name(value)
                        if pd.notna(value) and str(value).strip()
                        else f"Unnamed: {index}"
                        for index, value in enumerate(headers)
                    ]
                    df = raw_df.iloc[header_row + 1:].copy()
                    df.columns = columns
                    df = df.dropna(how="all").reset_index(drop=True)

            df.columns = [_normalize_col_name(c) for c in df.columns]

            # Promote subheader row 0 if columns contain Unnamed: and row 0 has table column tokens
            if any(str(c).startswith("Unnamed:") for c in df.columns) and not df.empty:
                row0 = df.iloc[0]
                row0_tokens = [
                    unicodedata.normalize("NFKD", str(v).casefold()).strip()
                    for v in row0
                    if pd.notna(v) and str(v).strip()
                ]
                match_count = sum(
                    1 for v in row0_tokens
                    if any(
                        tok in v
                        for tok in (
                            "so_luong", "unit", "donvi", "tri gia", "status", "gia le",
                            "gia", "ma hoa don", "ma phieu", "ngay giao dich", "khach",
                            "tenkh", "tien", "so luong hang", "ten sp", "gio", "ngay lap"
                        )
                    )
                )
                if match_count >= 2:
                    new_cols = []
                    aliases_to_add = {}
                    for c, v in zip(df.columns, row0):
                        v_str = str(v).strip() if pd.notna(v) else ""
                        c_str = str(c).strip()
                        c_compact = _compact_col_name(c_str)
                        if v_str and (c_str.startswith("Unnamed:") or c_compact in {"hang", "don", "tien"}):
                            clean_v = _normalize_col_name(v_str)
                            new_cols.append(clean_v)
                            if c_str.startswith("Unnamed:"):
                                aliases_to_add[c_str] = clean_v
                        else:
                            new_cols.append(_normalize_col_name(c_str))
                    df.columns = new_cols
                    df = df.iloc[1:].reset_index(drop=True)
                    # Preserve Unnamed: aliases for backward compatibility
                    for old_col, new_col in aliases_to_add.items():
                        if old_col not in df.columns:
                            df[old_col] = df[new_col]

            df = remove_subheader_and_subtotal_rows(df)
            df = _fill_merged_order_metadata(df)
            month = _monthly_sheet_period(sheet)
            default_year = int(month.split("-")[0]) if month else None
            df = _coerce_types(df, default_year=default_year)
            df = dedup_order_table_rows(df)
            # Use file_stem__sheet_name as key if multiple sheets
            key = f"{stem}__{sheet}" if len(xls.sheet_names) > 1 else stem
            result[key] = df
            if raw_stem != stem:
                raw_key = f"{raw_stem}__{sheet}" if len(xls.sheet_names) > 1 else raw_stem
                result[raw_key] = df
            if month is not None:
                monthly_sheets.append((sheet, month, df))
            logger.info("Loaded XLSX sheet '%s' from %s: %d rows × %d cols",
                        sheet, filepath, len(df), len(df.columns))
        except Exception as e:
            logger.warning("Failed to load sheet '%s' from %s: %s", sheet, filepath, e)

    if len(monthly_sheets) > 1:
        columns = tuple(monthly_sheets[0][2].columns)
        if all(tuple(frame.columns) == columns for _, _, frame in monthly_sheets):
            month_column = "__source_month"
            sheet_column = "__source_sheet"
            while month_column in columns or sheet_column in columns:
                month_column += "_"
                sheet_column += "_"
            combined = []
            for sheet, month, frame in monthly_sheets:
                month_frame = frame.copy()
                month_frame[month_column] = month
                month_frame[sheet_column] = sheet
                combined.append(month_frame)
            combined_table = pd.concat(combined, ignore_index=True)
            result[f"{stem}__monthly_combined"] = combined_table
            if raw_stem != stem:
                result[f"{raw_stem}__monthly_combined"] = combined_table
            logger.info(
                "Combined %d compatible monthly sheets from %s: %d rows",
                len(combined),
                filepath,
                len(combined_table),
            )
    return result


def _find_xlsx_header_row(df: pd.DataFrame, scan_rows: int = 50) -> int:
    """Find a likely header row in spreadsheets with report-title preambles."""
    best_row = 0
    best_score = (0, 0)

    for row_index, row in df.head(scan_rows).iterrows():
        values = [
            value for value in row.tolist()
            if pd.notna(value) and str(value).strip()
        ]
        if len(values) < 2:
            continue

        text_values = [value for value in values if isinstance(value, str)]
        if len(text_values) < 0.8 * len(values):
            continue

        unique_count = len({
            _normalize_col_name(value).casefold() for value in text_values
        })
        if unique_count < 0.8 * len(values):
            continue

        score = (len(values), unique_count)
        if score > best_score:
            best_row = int(row_index)
            best_score = score

    return best_row


def _load_sqlite(filepath: str) -> Dict[str, pd.DataFrame]:
    """Load all tables from a SQLite database, returning {table_name: DataFrame}."""
    result = {}
    conn = sqlite3.connect(filepath)
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        for table in tables:
            try:
                df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
                df.columns = [_normalize_col_name(c) for c in df.columns]
                df = remove_subheader_and_subtotal_rows(df)
                df = _coerce_types(df)
                df = dedup_order_table_rows(df)
                result[table] = df
                logger.info("Loaded SQLite table '%s' from %s: %d rows × %d cols",
                            table, filepath, len(df), len(df.columns))
            except Exception as e:
                logger.warning("Failed to load table '%s' from %s: %s", table, filepath, e)
    finally:
        conn.close()
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Cache to avoid reloading the same business data
_cache: Dict[str, Dict[str, pd.DataFrame]] = {}


def _combine_split_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Combine compatible split tables (e.g. _1 + _2 or _2025 + _2026 or monthly combined from split files)."""
    groups: Dict[Tuple[str, bool], List[Tuple[str, pd.DataFrame]]] = {}
    for name, df in list(tables.items()):
        m = re.match(r"^(.+?)_([0-9]{1,2}|20[0-9]{2})(?:__(monthly_combined.*))?$", name)
        if m:
            base_prefix = m.group(1)
            is_monthly = bool(m.group(3))
            key = (base_prefix, is_monthly)
            groups.setdefault(key, []).append((name, df))

    for (base_prefix, is_monthly), table_list in groups.items():
        if len(table_list) > 1:
            col0 = [c for c in table_list[0][1].columns if not str(c).startswith("__source_")]
            if all([c for c in t[1].columns if not str(c).startswith("__source_")] == col0 for t in table_list[1:]):
                combined_df = pd.concat([t[1] for t in table_list], ignore_index=True)
                if is_monthly:
                    comb_name = f"{base_prefix}__monthly_combined"
                    for part_name, _ in table_list:
                        tables.pop(part_name, None)
                    tables[comb_name] = combined_df
                else:
                    comb_name = f"{base_prefix}__combined"
                    tables[comb_name] = combined_df
                    if base_prefix not in tables:
                        tables[base_prefix] = combined_df
    return tables


def _bridge_relational_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Relational denormalization & dimension/temporal bridging across tables."""
    table_items = list(tables.items())

    # Pass 1: FK aliasing across tables
    for name_a, df_a in table_items:
        if df_a.empty:
            continue
        id_cols_a = [c for c in df_a.columns if _is_order_id_column(c)]
        for name_b, df_b in table_items:
            if df_a is df_b or name_a >= name_b or df_b.empty:
                continue
            id_cols_b = [c for c in df_b.columns if _is_order_id_column(c)]
            for ida in id_cols_a:
                set_a = set(df_a[ida].dropna().astype(str))
                if len(set_a) < 2:
                    continue
                for idb in id_cols_b:
                    if ida == idb:
                        continue
                    set_b = set(df_b[idb].dropna().astype(str))
                    if len(set_b) < 2:
                        continue
                    overlap = len(set_a & set_b)
                    if overlap >= 2 and (overlap / min(len(set_a), len(set_b))) >= 0.5:
                        if idb not in df_a.columns:
                            df_a[idb] = df_a[ida]
                        if ida not in df_b.columns:
                            df_b[ida] = df_b[idb]

    # Pass 2: Temporal propagation (Order/Invoice/Import Header -> Detail)
    for name_a, df_a in table_items:
        if df_a.empty:
            continue
        order_ids_a = [c for c in df_a.columns if _is_order_id_column(c)]
        date_cols_a = [
            c for c in df_a.columns
            if _is_date_column(c) and pd.api.types.is_datetime64_any_dtype(df_a[c])
        ]
        if not order_ids_a or not date_cols_a:
            continue

        for name_b, df_b in table_items:
            if df_a is df_b or name_a == name_b or df_b.empty:
                continue
            date_cols_b = [
                c for c in df_b.columns
                if _is_date_column(c) and pd.api.types.is_datetime64_any_dtype(df_b[c])
            ]
            if date_cols_b:
                continue

            for oid in order_ids_a:
                if oid in df_b.columns:
                    set_a = set(df_a[oid].dropna().astype(str))
                    set_b = set(df_b[oid].dropna().astype(str))
                    if len(set_a & set_b) >= 2 and (len(set_a & set_b) / max(1, len(set_b))) >= 0.5:
                        for dcol in date_cols_a:
                            if dcol not in df_b.columns:
                                mapping = (
                                    df_a.dropna(subset=[oid, dcol])
                                    .drop_duplicates(subset=[oid])
                                    .set_index(oid)[dcol]
                                )
                                df_b[dcol] = df_b[oid].map(mapping)

    # Pass 3: Product Dimension propagation (Catalog -> Detail/Secondary)
    sku_tokens = ("sku", "mahang", "masp", "mathuoc", "mahh")
    name_tokens = ("tensp", "tenhang", "tenthuoc", "tensanpham", "tenhh")
    hoat_chat_tokens = ("hoatchat", "activeingredient", "hoatchatchinh")
    ke_don_tokens = ("kedon", "thuocketheodon", "thuockedon", "bantheodon", "prescription", "theodon")
    nhom_hang_tokens = ("nhomhang", "danhmuc", "category", "nhomsanpham", "maloaihang", "tenloaihang", "loaihang")
    dvt_tokens = ("donvitinh", "dvt", "unit")

    for name_a, df_a in table_items:
        if df_a.empty:
            continue
        sku_cols_a = [c for c in df_a.columns if any(k in _compact_col_name(c) for k in sku_tokens)]
        name_cols_a = [c for c in df_a.columns if any(k in _compact_col_name(c) for k in name_tokens)]
        dim_cols_a = [
            c for c in df_a.columns
            if any(k in _compact_col_name(c) for k in (name_tokens + hoat_chat_tokens + ke_don_tokens + nhom_hang_tokens + dvt_tokens))
        ]
        if not dim_cols_a or (not sku_cols_a and not name_cols_a):
            continue

        for name_b, df_b in table_items:
            if df_a is df_b or name_a == name_b or df_b.empty:
                continue

            join_pairs = []
            for sa in sku_cols_a:
                for sb in [c for c in df_b.columns if any(k in _compact_col_name(c) for k in sku_tokens)]:
                    join_pairs.append((sa, sb))
            if not join_pairs:
                for na in name_cols_a:
                    for nb in [c for c in df_b.columns if any(k in _compact_col_name(c) for k in name_tokens)]:
                        join_pairs.append((na, nb))

            for key_a, key_b in join_pairs:
                set_a = set(df_a[key_a].dropna().astype(str))
                set_b = set(df_b[key_b].dropna().astype(str))
                if len(set_a & set_b) >= 2 and (len(set_a & set_b) / max(1, len(set_b))) >= 0.5:
                    for dcol in dim_cols_a:
                        if dcol != key_a and dcol not in df_b.columns:
                            mapping = (
                                df_a.dropna(subset=[key_a, dcol])
                                .drop_duplicates(subset=[key_a])
                                .set_index(key_a)[dcol]
                            )
                            mapped_series = df_b[key_b].map(mapping)
                            if any(k in _compact_col_name(dcol) for k in ke_don_tokens):
                                mapped_series = mapped_series.fillna(False)
                            df_b[dcol] = mapped_series

    # Pass 4: Customer dimension propagation (Customer Catalog -> Orders/Invoices)
    cust_id_tokens = ("makh", "makhach", "customerid", "makhachhang")
    cust_dim_tokens = ("tenkh", "tenkhach", "khachhangten", "customername", "nhomkhach", "nhomkh", "loaihinhkhach")
    for name_a, df_a in table_items:
        if df_a.empty:
            continue
        cust_id_cols_a = [
            c for c in df_a.columns
            if any(k in _compact_col_name(c) for k in cust_id_tokens)
        ]
        cust_dim_cols_a = [
            c for c in df_a.columns
            if any(k in _compact_col_name(c) for k in cust_dim_tokens)
        ]
        if not cust_id_cols_a or not cust_dim_cols_a:
            continue

        for name_b, df_b in table_items:
            if df_a is df_b or name_a == name_b or df_b.empty:
                continue

            cust_id_cols_b = [
                c for c in df_b.columns
                if any(k in _compact_col_name(c) for k in cust_id_tokens)
            ]
            for cida in cust_id_cols_a:
                for cidb in cust_id_cols_b:
                    set_a = set(df_a[cida].dropna().astype(str))
                    set_b = set(df_b[cidb].dropna().astype(str))
                    if len(set_a & set_b) >= 2 and (len(set_a & set_b) / max(1, len(set_b))) >= 0.5:
                        if cida not in df_b.columns:
                            df_b[cida] = df_b[cidb]
                        if cidb not in df_a.columns:
                            df_a[cidb] = df_a[cida]
                        for cdim in cust_dim_cols_a:
                            if cdim not in df_b.columns:
                                mapping = (
                                    df_a.dropna(subset=[cida, cdim])
                                    .drop_duplicates(subset=[cida])
                                    .set_index(cida)[cdim]
                                )
                                df_b[cdim] = df_b[cidb].map(mapping)

    # Pass 5: Pre-compute 'Số lượng quy đổi' (base-unit quantity) from unit conversion tables (Rulebook §3 & §6)
    ratio_tokens = ("tilequydoi", "heso", "hesoqd", "tylequydoi")
    large_unit_tokens = ("donviquydoi", "dvnhap", "dvlon", "donvilon", "dvtquydoi")
    prod_key_tokens = ("tensanpham", "tenhang", "tenmon", "hang", "tenhh", "tenhanghoa", "masp", "mahang", "masku", "sku")
    conv_map: Dict[Tuple[str, str], float] = {}
    for name_a, df_a in table_items:
        if df_a.empty:
            continue
        r_col = next((c for c in df_a.columns if _compact_col_name(c) in ratio_tokens), None)
        u_col = next((c for c in df_a.columns if _compact_col_name(c) in large_unit_tokens), None)
        p_cols = [c for c in df_a.columns if _compact_col_name(c) in prod_key_tokens]
        if r_col and u_col and p_cols:
            for p_col in p_cols:
                sub = df_a[[p_col, u_col, r_col]].dropna()
                for p_val, u_val, r_val in zip(sub[p_col], sub[u_col], sub[r_col]):
                    try:
                        f_val = float(r_val)
                        if f_val > 0:
                            conv_map[(normalize_text(p_val), normalize_text(u_val))] = f_val
                    except (ValueError, TypeError):
                        pass

    if conv_map:
        qty_col_candidates = ("Số lượng", "Số lượng sản phẩm", "so_luong", "SL")
        unit_col_candidates = ("ĐVT", "Đơn vị tính", "Đơn vị", "Unit", "dvt", "don_vi_tinh")
        for name_b, df_b in table_items:
            if df_b.empty or "Số lượng quy đổi" in df_b.columns or "SL quy đổi" in df_b.columns:
                continue
            if _compact_col_name(name_b) in ("danhsachquydoi", "banhangdonviquydoi", "danhsachdonvitinh", "donvisp", "danhmuc"):
                continue
            q_col = next((c for c in qty_col_candidates if c in df_b.columns), None)
            u_col = next((c for c in unit_col_candidates if c in df_b.columns), None)
            p_cols_b = [c for c in df_b.columns if _compact_col_name(c) in prod_key_tokens]
            if q_col and u_col and p_cols_b:
                p_col_b = p_cols_b[0]
                p_norm = df_b[p_col_b].fillna("").astype(str).map(normalize_text)
                u_norm = df_b[u_col].fillna("").astype(str).map(normalize_text)
                factors = [
                    conv_map.get((p, u), 1.0)
                    for p, u in zip(p_norm, u_norm)
                ]
                q_num = pd.to_numeric(df_b[q_col], errors="coerce")
                df_b["Số lượng quy đổi"] = q_num * pd.Series(factors, index=df_b.index)

    return tables


def load_business_data(business_dir: str, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load all data files from a business directory.

    Parameters
    ----------
    business_dir : str
        Path to the business data directory (e.g., /data/data/biz_017/).
    use_cache : bool
        If True, cache and reuse previously loaded data for this directory.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Mapping from table/sheet/file name to DataFrame.
    """
    business_dir = str(Path(business_dir).resolve())

    if use_cache and business_dir in _cache:
        logger.debug("Cache hit for %s", business_dir)
        return _cache[business_dir]

    tables: Dict[str, pd.DataFrame] = {}
    data_dir = Path(business_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Business directory not found: {business_dir}")

    for f in sorted(data_dir.iterdir()):
        if f.name.startswith(".") or f.name == "README.md":
            continue

        ext = f.suffix.lower()
        try:
            if ext == ".csv":
                tables.update(_load_csv(str(f)))
            elif ext in (".xlsx", ".xls"):
                tables.update(_load_xlsx(str(f)))
            elif ext in (".sqlite", ".db", ".sqlite3"):
                tables.update(_load_sqlite(str(f)))
            else:
                logger.debug("Skipping unsupported file: %s", f)
        except Exception as e:
            logger.error("Failed to load %s: %s", f, e, exc_info=True)

    if not tables:
        logger.warning("No data tables loaded from %s", business_dir)
    else:
        # Check README.md for unit conversions if no conversion table exists
        readme_file = data_dir / "README.md"
        if readme_file.exists():
            try:
                content = readme_file.read_text(encoding="utf-8", errors="ignore")
                has_conv_table = any(
                    "quydoi" in _compact_col_name(tname)
                    for tname in tables
                )
                if not has_conv_table:
                    matches = re.findall(
                        r"-\s+([^:\n]+):\s*1\s+([^=\n]+)\s*=\s*(\d+(?:[.,]\d+)?)\s*([^\n\r]+)",
                        content,
                    )
                    if matches:
                        rows = []
                        for p_name, u_from, ratio_str, u_to in matches:
                            ratio_val = float(ratio_str.replace(",", "."))
                            if ratio_val.is_integer():
                                ratio_val = int(ratio_val)
                            rows.append({
                                "Tên sản phẩm": p_name.strip(),
                                "Tên hàng": p_name.strip(),
                                "Đơn vị quy đổi": u_from.strip(),
                                "Tỉ lệ quy đổi": ratio_val,
                                "Đơn vị cơ bản": u_to.strip(),
                            })
                        df_conv = pd.DataFrame(rows)
                        df_conv = _coerce_types(df_conv)
                        tables["DanhSachQuyDoi"] = df_conv
                        tables["danh_sach_quy_doi"] = df_conv
            except Exception as e:
                logger.warning("Failed to parse conversion from README.md: %s", e)

        tables = _combine_split_tables(tables)
        tables = _bridge_relational_tables(tables)

    if use_cache:
        _cache[business_dir] = tables

    return tables


def clear_cache():
    """Clear the data cache."""
    _cache.clear()


def get_table_info(tables: Dict[str, pd.DataFrame]) -> str:
    """
    Generate a human-readable summary of all loaded tables,
    suitable for inclusion in an LLM prompt.
    """
    lines = []
    for name, df in tables.items():
        lines.append(f"### Table: {name}")
        lines.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        lines.append("")
        lines.append("| Column | Type | Non-null | Unique | Sample values |")
        lines.append("|--------|------|----------|--------|---------------|")
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = df[col].notna().sum()
            nunique = df[col].nunique()
            samples = df[col].dropna().head(3).tolist()
            sample_str = ", ".join(str(s) for s in samples)
            if len(sample_str) > 60:
                sample_str = sample_str[:57] + "..."
            lines.append(f"| {col} | {dtype} | {non_null} | {nunique} | {sample_str} |")
        lines.append("")
    return "\n".join(lines)
