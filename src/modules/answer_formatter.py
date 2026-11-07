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
    """Format a percentage answer. Rounded to 2 decimal places."""
    if value is None:
        return None
    try:
        num = float(value)
        if math.isnan(num) or math.isinf(num):
            return None
        return round(num, 2)
    except (ValueError, TypeError):
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


def format_list(value: Any) -> Optional[List]:
    """Format a list answer."""
    if value is None:
        return None

    # Convert various types to list
    if isinstance(value, pd.Series):
        value = value.tolist()
    elif isinstance(value, pd.Index):
        value = value.tolist()
    elif isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [v.strip() for v in value.split(",") if v.strip()]

    if not isinstance(value, list):
        value = [value]

    nested_types = (
        list, tuple, dict, set, np.ndarray, pd.Series, pd.Index, pd.DataFrame
    )
    if any(isinstance(item, nested_types) for item in value):
        return None

    # Format each element
    formatted = []
    for item in value:
        if isinstance(item, (int, float)):
            formatted.append(format_number(item))
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
            if (question_is_average_order_value(question) or question_is_monetary_average(question)) and isinstance(formatted_value, (int, float)):
                formatted_value = round_half_up_vnd(formatted_value)
        except Exception as e:
            logger.warning("Error in monetary average rounding: %s", e)

    if formatted_value is None:
        logger.warning("Formatting produced None for type=%s, value=%r",
                       answer_type, value)
        return {"type": "abstain", "reason": "formatting_failure"}

    return {"type": answer_type, "answer": formatted_value}


def build_submission_line(question_id: str, answer: Dict[str, Any]) -> str:
    """Build a single JSONL line for submission."""
    entry = {"id": question_id}
    entry.update(answer)
    return json.dumps(entry, ensure_ascii=False)
