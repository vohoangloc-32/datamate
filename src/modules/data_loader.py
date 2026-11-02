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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_col_name(name: str) -> str:
    """Normalize a column name: Unicode NFC, strip, collapse whitespace."""
    name = unicodedata.normalize("NFC", str(name))
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
    return any(token in compact for token in ("madon", "orderid", "orderno", "ordernumber"))


def _is_order_metadata_column(name: str) -> bool:
    compact = _compact_col_name(name)
    return (
        _is_order_id_column(name)
        or any(token in compact for token in (
            "ngaydat",
            "orderdate",
            "trangthaidonhang",
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


def _coerce_date_column(values: pd.Series) -> Optional[pd.Series]:
    """Parse date-like columns, including Excel serial-day values."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values

    non_null = values.dropna()
    if non_null.empty:
        return None

    numeric = pd.to_numeric(values, errors="coerce")
    valid_numeric = numeric.dropna()
    if not valid_numeric.empty:
        if valid_numeric.between(20000, 80000).all():
            parsed_excel_dates = pd.to_datetime(
                numeric,
                unit="D",
                origin="1899-12-30",
                errors="coerce",
            )
            if parsed_excel_dates.notna().any():
                return parsed_excel_dates
        if valid_numeric.notna().sum() == len(non_null):
            return None

    date_text = values.astype("string").str.strip()
    date_text = date_text.str.replace(
        r"(?i)\bth(?:áng|g)\.?\s*",
        "",
        regex=True,
    )
    parsed_dates = pd.to_datetime(date_text, errors="coerce", dayfirst=True)
    iso_dates = date_text.str.match(r"^\d{4}-\d{1,2}-\d{1,2}(?:\b|T)", na=False)
    if iso_dates.any():
        parsed_dates.loc[iso_dates] = pd.to_datetime(
            date_text.loc[iso_dates],
            errors="coerce",
            yearfirst=True,
        )
    if parsed_dates.notna().any():
        return parsed_dates
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


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Try to coerce object/string columns to numeric or datetime where appropriate."""
    for col in df.columns:
        if _is_date_column(col):
            parsed_dates = _coerce_date_column(df[col])
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
                except Exception:
                    pass
    return df


# ---------------------------------------------------------------------------
# Loaders per format
# ---------------------------------------------------------------------------

def _load_csv(filepath: str) -> Dict[str, pd.DataFrame]:
    """Load a CSV file, returning {filename_stem: DataFrame}."""
    stem = Path(filepath).stem
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
    df = _fill_merged_order_metadata(df)
    df = _coerce_types(df)
    logger.info("Loaded CSV %s: %d rows × %d cols", filepath, len(df), len(df.columns))
    return {stem: df}


def _load_xlsx(filepath: str) -> Dict[str, pd.DataFrame]:
    """Load an Excel file (all sheets), returning {sheet_name: DataFrame}."""
    result = {}
    monthly_sheets = []
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
            df = _fill_merged_order_metadata(df)
            df = _coerce_types(df)
            # Use file_stem__sheet_name as key if multiple sheets
            key = f"{Path(filepath).stem}__{sheet}" if len(xls.sheet_names) > 1 else Path(filepath).stem
            result[key] = df
            month = _monthly_sheet_period(sheet)
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
            result[f"{Path(filepath).stem}__monthly_combined"] = combined_table
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
                df = _coerce_types(df)
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
