"""
Tests for answer_formatter module.
"""

import json
import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime

from src.modules.answer_formatter import (
    normalize_string,
    format_number,
    format_percent,
    format_string,
    format_date,
    format_boolean,
    format_list,
    format_table,
    format_answer,
    build_submission_line,
)


def test_normalize_string():
    # Lowercase + strip + space collapse
    assert normalize_string("  Cà  phê   sữa đá  ") == "cà phê sữa đá"
    # Punctuation to space
    assert normalize_string("Đơn hàng #123! [VIP]") == "đơn hàng 123 vip"
    # Preserves diacritics
    assert normalize_string("Hà Nội - TP. Hồ Chí Minh") == "hà nội tp hồ chí minh"
    # Empty or symbols only
    assert normalize_string("---***---") == ""


def test_format_number():
    assert format_number(42) == 42
    assert format_number(42.0) == 42
    assert format_number(12.3456) == 12.35
    assert format_number("100") == 100
    assert format_number("100.555") == 100.56
    assert format_number(None) is None
    assert format_number(float("nan")) is None
    assert format_number(float("inf")) is None


def test_format_percent():
    assert format_percent(15.678) == 15.68
    assert format_percent(-5.556) == -5.56
    assert format_percent("20.1") == 20.1
    assert format_percent(None) is None


def test_format_string():
    assert format_string("  Quán Cà Phê Mùa Thu  ") == "Quán Cà Phê Mùa Thu"
    assert format_string(None) is None
    assert format_string("") is None
    assert format_string(123) == "123"


def test_format_date():
    assert format_date("2024-05-18") == "2024-05-18"
    assert format_date("2024-05") == "2024-05"
    assert format_date("18/05/2024") == "2024-05-18"
    assert format_date(date(2024, 5, 18)) == "2024-05-18"
    assert format_date(datetime(2024, 5, 18, 10, 30)) == "2024-05-18"
    assert format_date(pd.Timestamp("2024-05-18")) == "2024-05-18"
    assert format_date("invalid_date") is None


def test_format_boolean():
    assert format_boolean(True) is True
    assert format_boolean(False) is False
    assert format_boolean("true") is True
    assert format_boolean("false") is False
    assert format_boolean("đúng") is True
    assert format_boolean("sai") is False
    assert format_boolean("có") is True
    assert format_boolean("không") is False
    assert format_boolean(1) is True
    assert format_boolean(0) is False
    assert format_boolean("unrecognized") is None


def test_format_list():
    assert format_list(["A", "B", "C"]) == ["A", "B", "C"]
    assert format_list([1, 2.5, 3]) == [1, 2.5, 3]
    assert format_list(pd.Series(["X", "Y"])) == ["X", "Y"]
    assert format_list(np.array([10, 20])) == [10, 20]
    assert format_list('["apple", "banana"]') == ["apple", "banana"]
    assert format_list("single_string") == ["single_string"]


@pytest.mark.parametrize(
    "value",
    [
        [["2026-03", 5827400]],
        [{"month": "2026-03"}],
        '[[\"2026-03\", 5827400]]',
        np.array([["2026-03", 5827400]], dtype=object),
    ],
)
def test_format_list_rejects_nested_elements(value):
    assert format_list(value) is None
    assert format_answer(value, "list") == {
        "type": "abstain",
        "reason": "formatting_failure",
    }


def test_format_table():
    df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
    res = format_table(df)
    assert res == {
        "columns": ["col_a", "col_b"],
        "rows": [[1, "x"], [2, "y"]],
    }
    custom_dict = {"columns": ["name"], "rows": [["Alice"], ["Bob"]]}
    assert format_table(custom_dict) == custom_dict
    assert format_table(None) is None


def test_format_table_with_pandas_period_values():
    period = pd.Period("2025-02", freq="M")
    answer = format_answer(
        {"columns": ["month", "revenue"], "rows": [[period, np.int64(1200)]]},
        "table",
    )

    assert answer == {
        "type": "table",
        "answer": {"columns": ["month", "revenue"], "rows": [["2025-02", 1200]]},
    }
    parsed = json.loads(build_submission_line("q_period", answer))
    assert parsed["answer"]["rows"][0][0] == "2025-02"


def test_format_answer():
    # Number
    ans = format_answer(123.456, "number")
    assert ans == {"type": "number", "answer": 123.46}

    # Percent
    ans = format_answer(10.5, "percent")
    assert ans == {"type": "percent", "answer": 10.5}

    # Abstain
    ans = format_answer(None, "abstain", reason="low_confidence")
    assert ans == {"type": "abstain", "reason": "low_confidence"}

    # Unanswerable
    ans = format_answer(None, "unanswerable", reason="missing_column")
    assert ans == {"type": "unanswerable", "reason": "missing_column"}

    # Formatting failure -> abstain
    ans = format_answer(None, "number")
    assert ans == {"type": "abstain", "reason": "formatting_failure"}


def test_build_submission_line():
    ans = {"type": "number", "answer": 500}
    line = build_submission_line("q_001", ans)
    parsed = json.loads(line)
    assert parsed == {"id": "q_001", "type": "number", "answer": 500}
