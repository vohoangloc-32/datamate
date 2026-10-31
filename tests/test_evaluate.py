"""
Tests for evaluate module (Penalized Accuracy Scorer).
"""

import pytest
from src.evaluate import (
    normalize_string,
    match_number,
    match_percent,
    match_string,
    match_date,
    match_boolean,
    match_list,
    match_table,
    score_answer,
    compute_pa,
)


def test_match_number():
    assert match_number(100, 100)
    assert match_number(12.345, 12.348)  # < 0.005 diff
    assert not match_number(12.34, 12.35)
    assert not match_number("not_a_number", 100)


def test_match_percent():
    assert match_percent(15.25, 15.25)
    assert match_percent(15.252, 15.254)  # within 0.005 after rounding
    assert not match_percent(15.25, 16.25)


def test_match_string():
    assert match_string("Cà phê sữa đá", "cà phê sữa đá")
    assert match_string("Quán  A (Chi nhánh 1) ", "quán a chi nhánh 1")
    # Matching with aliases
    assert match_string("Gà Rán", "Gà chiên", aliases=["Gà rán", "Fried Chicken"])
    assert not match_string("Trà đào", "Cà phê sữa đá")


def test_match_date():
    assert match_date("2024-05-18", "2024-05-18")
    assert match_date("2024-05", "2024-05")
    assert not match_date("2024-05-18", "2024-05-19")


def test_match_boolean():
    assert match_boolean(True, True)
    assert match_boolean(False, False)
    assert match_boolean("True", True)
    assert match_boolean(1, True)
    assert not match_boolean(True, False)


def test_match_list():
    assert match_list(["A", "B", "C"], ["a", "b", "c"])
    assert match_list([10, 20.0], [10.0, 20])
    # Order matters
    assert not match_list(["A", "B"], ["B", "A"])
    # Length mismatch
    assert not match_list(["A"], ["A", "B"])


def test_match_table():
    # Identical table
    pred = {"columns": ["name", "score"], "rows": [["Alice", 90], ["Bob", 85]]}
    ref = {"columns": ["name", "score"], "rows": [["Alice", 90], ["Bob", 85]]}
    assert match_table(pred, ref)

    # Columns in different order
    pred_reordered_cols = {"columns": ["score", "name"], "rows": [[90, "Alice"], [85, "Bob"]]}
    assert match_table(pred_reordered_cols, ref)

    # Rows in different order (rows treated as set)
    pred_reordered_rows = {"columns": ["name", "score"], "rows": [["Bob", 85], ["Alice", 90]]}
    assert match_table(pred_reordered_rows, ref)

    # Content mismatch
    pred_wrong = {"columns": ["name", "score"], "rows": [["Alice", 90], ["Bob", 50]]}
    assert not match_table(pred_wrong, ref)


def test_score_answer():
    # Correct (+1)
    assert score_answer(
        {"type": "number", "answer": 100},
        {"type": "number", "answer": 100}
    ) == 1

    # Abstain (0)
    assert score_answer(
        {"type": "abstain"},
        {"type": "number", "answer": 100}
    ) == 0

    # Wrong (-1)
    assert score_answer(
        {"type": "number", "answer": 99},
        {"type": "number", "answer": 100}
    ) == -1

    # Type mismatch (-1)
    assert score_answer(
        {"type": "string", "answer": "100"},
        {"type": "number", "answer": 100}
    ) == -1

    # Unanswerable correct (+1)
    assert score_answer(
        {"type": "unanswerable"},
        {"type": "unanswerable"}
    ) == 1

    # Unanswerable when answerable (-1)
    assert score_answer(
        {"type": "unanswerable"},
        {"type": "number", "answer": 50}
    ) == -1


def test_compute_pa():
    predictions = {
        "q1": {"type": "number", "answer": 10},  # Correct (+1)
        "q2": {"type": "number", "answer": 5},   # Wrong (-1)
        "q3": {"type": "abstain"},               # Abstain (0)
        # q4 is missing -> Abstain (0)
    }
    references = {
        "q1": {"type": "number", "answer": 10},
        "q2": {"type": "number", "answer": 20},
        "q3": {"type": "string", "answer": "test"},
        "q4": {"type": "boolean", "answer": True},
    }

    results = compute_pa(predictions, references)

    # Total = 4, Correct = 1, Wrong = 1, Abstain = 2 (1 explicit + 1 missing)
    assert results["total"] == 4
    assert results["correct"] == 1
    assert results["wrong"] == 1
    assert results["abstain"] == 2
    # PA = (1 - 1) / 4 = 0.0
    assert results["penalized_accuracy"] == 0.0
