"""
answer_formatter.py — Answer Formatting & Validation for DataMate

Ensures every answer matches the exact submission format required by
the competition. Handles all answer types: number, percent, string,
date, boolean, list, table, unanswerable, abstain.
"""

import json
import logging
import math
import re
import unicodedata
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# String normalization (matches competition evaluation)
# ---------------------------------------------------------------------------

def normalize_string(s: str) -> str:
    """
    Normalize a string for comparison, following the competition rules:
    1. Unicode NFC
    2. Lowercase
    3. Every character that is not a letter, digit, or whitespace → space
    4. Leading/trailing whitespace removed
    5. Runs of whitespace collapsed to one space
    Note: Diacritics are NOT removed.
    """
    s = unicodedata.normalize("NFC", str(s))
    s = s.lower()
    # Replace non-letter, non-digit, non-whitespace with space
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


# ---------------------------------------------------------------------------
# Type-specific formatters
# ---------------------------------------------------------------------------

def format_number(value: Any) -> Optional[Union[int, float]]:
    """Format a numeric answer. Returns integer if exact, else rounded to 2 decimals."""
    if value is None:
        return None
    try:
        num = float(value)
        if math.isnan(num) or math.isinf(num):
            return None
        # Check if it's effectively an integer
        if num == int(num) and abs(num) < 1e15:
            return int(num)
        return round(num, 2)
    except (ValueError, TypeError):
        return None


def format_percent(value: Any) -> Optional[float]:
    """Format a percentage answer. Rounded to 2 decimal places using ROUND_HALF_UP."""
    if value is None:
        return None
    try:
        num = float(value)
        if math.isnan(num) or math.isinf(num):
            return None
        from decimal import Decimal, ROUND_HALF_UP
        d = Decimal(str(num)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return float(d)
    except (ValueError, TypeError):
        try:
            return round(float(value), 2)
        except Exception:
            return None


def format_string(value: Any) -> Optional[str]:
    """Format a string answer."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Unicode NFC normalize but don't lowercase (preserve original casing)
    return unicodedata.normalize("NFC", s)


def format_date(value: Any) -> Optional[str]:
    """Format a date answer in ISO format (YYYY-MM-DD or YYYY-MM)."""
    if value is None:
        return None

    s = str(value).strip()

    # Already in ISO format?
    if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        return s
    if re.match(r"^\d{4}-\d{2}$", s):
        return s

    # Try parsing various formats
    formats = [
        "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
        "%Y/%m/%d", "%d-%m-%Y", "%Y-%m",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            if fmt == "%Y-%m":
                return dt.strftime("%Y-%m")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Handle pandas Timestamp
    if isinstance(value, (pd.Timestamp, datetime, date)):
        if isinstance(value, date) and not isinstance(value, datetime):
            return value.isoformat()
        return value.strftime("%Y-%m-%d")

    return None


def format_boolean(value: Any) -> Optional[bool]:
    """Format a boolean answer."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("true", "yes", "đúng", "có", "1"):
        return True
    if s in ("false", "no", "sai", "không", "0"):
        return False
    return None


def _is_pure_numeric_str(s: str) -> bool:
    """True when string is a clean finite JSON number (without leading-zero phone/ID format)."""
    return bool(re.match(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?$", s.strip()))


def format_list(value: Any) -> Optional[List]:
    """Format a list answer."""
    if value is None:
        return None

    # Convert various types to list
    if isinstance(value, (tuple, set)):
        value = list(value)
    elif isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0].tolist() if not value.empty else []
    elif isinstance(value, pd.Series):
        value = value.tolist()
    elif isinstance(value, pd.Index):
        value = value.tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, dict):
        value = list(value.values()) if all(isinstance(v, (int, float, np.integer, np.floating)) for v in value.values()) else list(value.keys())
    elif isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [v.strip() for v in value.split(",") if v.strip()]

    if not isinstance(value, list):
        value = [value]

    # Flatten 1D-like nested structures: [[x], [y], [z]] -> [x, y, z]
    if any(isinstance(item, (list, tuple)) for item in value):
        flattened = []
        is_flattenable = True
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 1:
                flattened.append(item[0])
            elif not isinstance(item, (list, tuple, dict, set, np.ndarray, pd.Series, pd.Index, pd.DataFrame)):
                flattened.append(item)
            else:
                is_flattenable = False
                break
        if is_flattenable:
            value = flattened

    nested_types = (
        list, tuple, dict, set, np.ndarray, pd.Series, pd.Index, pd.DataFrame
    )
    if any(isinstance(item, nested_types) for item in value):
        return None

    # Unpack 2-part mixed unit strings (e.g. ['10882 vỉ chẵn', '0 viên lẻ'] or ['nợ thêm 80000', 'trả được 40000'])
    if len(value) == 2 and all(isinstance(x, str) for x in value):
        # 1. Clean Vietnamese thousands separators first
        clean0 = str(value[0]).replace(".", "").replace(",", ".")
        clean1 = str(value[1]).replace(".", "").replace(",", ".")
        m0 = re.search(r"(-?(?:0|[1-9]\d*)(?:\.\d+)?)", clean0)
        m1 = re.search(r"(-?(?:0|[1-9]\d*)(?:\.\d+)?)", clean1)
        joined_lower = f"{value[0]} {value[1]}".lower()
        if m0 and m1 and (
            any(k in joined_lower for k in ("chẵn", "lẻ", "chan", "le", "xe 4m3", "cuộn 10m", "vỉ", "viên", "hộp", "thùng", "m3", "nợ", "no", "tra", "trả", "them", "thêm"))
            or not (_is_pure_numeric_str(value[0]) and _is_pure_numeric_str(value[1]))
        ):
            return [format_number(m0.group(1)), format_number(m1.group(1))]

    # Format each element
    formatted = []
    for item in value:
        if isinstance(item, (int, float, np.integer, np.floating)) and not isinstance(item, bool):
            formatted.append(format_number(item))
        elif isinstance(item, str) and _is_pure_numeric_str(item):
            formatted.append(format_number(item.strip()))
        else:
            formatted.append(format_string(item))

    return formatted


def format_table(value: Any) -> Optional[Dict]:
    """
    Format a table answer as {"columns": [...], "rows": [[...], ...]}.
    """
    if value is None:
        return None

    if isinstance(value, pd.DataFrame):
        return {
            "columns": [str(c) for c in value.columns.tolist()],
            "rows": _to_json_compatible(value.values.tolist()),
        }

    if isinstance(value, dict):
        if "columns" in value and "rows" in value:
            return _to_json_compatible(value)

    return None


def _to_json_compatible(value: Any) -> Any:
    """Convert common pandas and NumPy values to JSON-compatible values."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Period):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _to_json_compatible(value.item())
    if isinstance(value, np.ndarray):
        return _to_json_compatible(value.tolist())
    if isinstance(value, (pd.Series, pd.Index)):
        return _to_json_compatible(value.tolist())
    if isinstance(value, dict):
        return {
            _to_json_compatible(key): _to_json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"Unsupported value for JSON serialization: {type(value).__name__}"
    )


# ---------------------------------------------------------------------------
# Main formatter
# ---------------------------------------------------------------------------

_FORMATTERS = {
    "number": format_number,
    "percent": format_percent,
    "string": format_string,
    "date": format_date,
    "boolean": format_boolean,
    "list": format_list,
    "table": format_table,
}


def format_answer(
    value: Any,
    answer_type: str,
    reason: Optional[str] = None,
    question: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Format an answer for submission.

    Parameters
    ----------
    value : Any
        The raw answer value.
    answer_type : str
        The answer type string.
    reason : str, optional
        Reason for unanswerable/abstain.
    question : str, optional
        The question text to apply rulebook domain rounding (e.g., AOV integer rounding).

    Returns
    -------
    Dict[str, Any]
        Formatted submission entry (without "id" field).
    """
    if answer_type == "abstain":
        entry = {"type": "abstain"}
        if reason:
            entry["reason"] = reason
        return entry

    if answer_type == "unanswerable":
        entry = {"type": "unanswerable"}
        if reason:
            entry["reason"] = reason
        return entry

    # Rescue type mismatches when question context is available (Rulebook §9)
    if question and value is not None:
        try:
            from .accounting_cleaner import normalize_text
            from .question_understanding import normalize_question_slang
            q_norm = normalize_text(normalize_question_slang(question))
            if answer_type == "string":
                if re.search(r"\bmay\b.*\bchan\b.*\bmay\b.*\ble\b", q_norm):
                    answer_type = "list"
                elif (isinstance(value, (int, float)) and not isinstance(value, bool)) or (
                    isinstance(value, str) and _is_pure_numeric_str(value)
                ):
                    if re.search(r"(?:%|phan tram|ty le|ti le)", q_norm):
                        answer_type = "percent"
                    else:
                        answer_type = "number"
            elif answer_type == "number":
                if re.search(r"(?:bao nhieu\s*%|%\s*(?:doanh thu|dthu|tong|ca cua hang)|dong gop\s+bao nhieu\s*%)", q_norm):
                    answer_type = "percent"
        except Exception as e:
            logger.debug("Type rescue skipped: %s", e)

    formatter = _FORMATTERS.get(answer_type)
    if formatter is None:
        logger.warning("Unknown answer type: %s, defaulting to string", answer_type)
        formatter = format_string
        answer_type = "string"

    formatted_value = _to_json_compatible(formatter(value))

    if formatted_value is not None and answer_type == "number" and question:
        try:
            from .accounting_cleaner import (
                question_is_average_order_value,
                question_is_monetary_average,
                round_half_up_vnd,
            )
            from .question_understanding import normalize_question_slang
            q_clean = normalize_question_slang(question)
            if (question_is_average_order_value(q_clean) or question_is_monetary_average(q_clean)) and isinstance(formatted_value, (int, float)):
                formatted_value = round_half_up_vnd(formatted_value)
        except Exception as e:
            logger.warning("Error in monetary average rounding: %s", e)

    if formatted_value is None:
        logger.warning("Formatting produced None for type=%s, value=%r",
                       answer_type, value)
        return {"type": "abstain", "reason": "formatting_failure"}

    return {"type": answer_type, "answer": formatted_value}


def resolve_entity_codes_in_answer(
    answer: Dict[str, Any],
    question: str,
    tables: Optional[Dict[str, pd.DataFrame]],
) -> Dict[str, Any]:
    """
    If a string, list, or table answer returns SKU/Customer IDs (e.g. 'SP0053', ['SP0001', ...])
    when the question asks for product/customer names (and not codes), map them to names.
    Also strips redundant code column from 3-column [Code, Name, Metric] tables.
    """
    if not answer or not question:
        return answer
    if tables is None:
        tables = {}
    ans_type = answer.get("type")
    if ans_type not in ("string", "list", "table"):
        return answer
    val = answer.get("answer")
    if not val:
        return answer

    try:
        from .accounting_cleaner import compact_identifier, normalize_text
        q_norm = normalize_text(question)
        if re.search(r"\bma\s+(?:hang|san pham|sp|sku|khach|kh|don|nhan vien|nv)\b", q_norm):
            return answer

        code_pattern = re.compile(r"^(?:SP|HH|MH|TH|KH|MHH|MSP|CUST)\d+$", re.IGNORECASE)
        sku_tokens = (
            "sku", "masku", "mahang", "masp", "masanpham", "mathuoc", "mahh",
            "mahanghoa", "makh", "makhach", "makhachhang", "customerid", "idsp", "idkh",
        )
        name_tokens = (
            "tensp", "tenhang", "tenthuoc", "tensanpham", "tenhh", "tenhanghoa",
            "tenmon", "tenkh", "tenkhach", "tenkhachhang", "khachhangten", "tengoi",
            "product", "hang", "sanpham", "khachhang",
        )

        code_to_name: Dict[str, str] = {}
        for df in tables.values():
            if df.empty:
                continue
            c_cols = [
                c for c in df.columns
                if compact_identifier(c) in sku_tokens
                or compact_identifier(c).startswith(("ma", "id", "code"))
                or any(code_pattern.match(str(x).strip()) for x in df[c].dropna().head(10))
            ]
            n_cols = [
                c for c in df.columns
                if compact_identifier(c) in name_tokens
                or compact_identifier(c) in ("kh", "ten", "hovaten", "customername", "tengoi")
                or any(t in compact_identifier(c) for t in ("ten", "name", "khachhang", "sanpham"))
            ]
            for cc in c_cols:
                for nc in n_cols:
                    if cc == nc:
                        continue
                    sub = df[[cc, nc]].dropna()
                    for k, v in zip(sub[cc].astype(str).str.strip(), sub[nc].astype(str).str.strip()):
                        if k and v and k not in code_to_name and code_pattern.match(k):
                            code_to_name[k] = v

        if ans_type == "table" and isinstance(val, dict):
            cols = val.get("columns", [])
            rows = val.get("rows", [])
            if cols and rows:
                c_comps = [compact_identifier(c) for c in cols]
                # If 3 columns: [Code, Name, Metric] -> drop Code column
                if len(cols) == 3:
                    if (
                        (c_comps[0] in sku_tokens or any(code_pattern.match(str(r[0]).strip()) for r in rows[:3] if r))
                        and (c_comps[1] in name_tokens or not str(rows[0][1]).replace(".", "").isdigit())
                    ):
                        new_cols = [cols[1], cols[2]]
                        new_rows = [[r[1], r[2]] for r in rows if len(r) >= 3]
                        return {"type": "table", "answer": {"columns": new_cols, "rows": new_rows}}
                # If 2 columns: [Code, Metric] -> resolve Code in rows to Name
                elif len(cols) == 2 and code_to_name:
                    if c_comps[0] in sku_tokens or any(code_pattern.match(str(r[0]).strip()) for r in rows[:3] if r):
                        new_rows = [
                            [code_to_name.get(str(r[0]).strip(), r[0]), r[1]]
                            for r in rows if len(r) >= 2
                        ]
                        return {"type": "table", "answer": {"columns": cols, "rows": new_rows}}

        if not code_to_name:
            return answer

        items = [val] if ans_type == "string" else (val if ans_type == "list" else [])
        if not any(isinstance(x, str) and (code_pattern.match(x.strip()) or x.strip() in code_to_name) for x in items):
            return answer

        if ans_type == "string" and isinstance(val, str):
            mapped = code_to_name.get(val.strip())
            if mapped:
                return {"type": "string", "answer": format_string(mapped)}
        elif ans_type == "list" and isinstance(val, list):
            new_list = [
                format_string(code_to_name.get(x.strip(), x)) if isinstance(x, str) else x
                for x in val
            ]
            return {"type": "list", "answer": new_list}
    except Exception as e:
        logger.debug("Entity code resolution skipped: %s", e)

    return answer


def build_submission_line(question_id: str, answer: Dict[str, Any]) -> str:
    """Build a single JSONL line for submission."""
    entry = {"id": question_id}
    entry.update(answer)
    return json.dumps(entry, ensure_ascii=False)
