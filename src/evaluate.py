"""
evaluate.py — Local PA Scorer for DataMate

Computes Penalized Accuracy (PA) on the training set by comparing
submission.jsonl against answers.jsonl.

Usage:
    python evaluate.py --submission submission.jsonl --answers data/train/answers.jsonl
"""

import argparse
import json
import math
import re
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Normalization (matches competition rules)
# ---------------------------------------------------------------------------

def normalize_string(s: str) -> str:
    """Normalize a string for comparison per competition rules."""
    s = unicodedata.normalize("NFC", str(s))
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


# ---------------------------------------------------------------------------
# Per-type matching
# ---------------------------------------------------------------------------

def match_number(pred: Any, ref: Any) -> bool:
    """Match numeric answers. Reference is integer or rounded to 2 decimals."""
    try:
        p = float(pred)
        r = float(ref)
        # Allow small floating-point tolerance
        return abs(p - r) < 0.005
    except (ValueError, TypeError):
        return False


def match_percent(pred: Any, ref: Any) -> bool:
    """Match percentage answers. Exact after rounding to 2 decimals."""
    try:
        p = round(float(pred), 2)
        r = round(float(ref), 2)
        return abs(p - r) < 0.005
    except (ValueError, TypeError):
        return False


def match_string(pred: Any, ref: Any, aliases: Optional[List[str]] = None) -> bool:
    """
    Match string answers after normalization.
    Checks against the reference and any aliases.
    """
    pred_norm = normalize_string(str(pred))

    accepted = [normalize_string(str(ref))]
    if aliases:
        for alias in aliases:
            accepted.append(normalize_string(str(alias)))

    return pred_norm in accepted


def match_date(pred: Any, ref: Any) -> bool:
    """Match date answers. Exact match in ISO format."""
    return str(pred).strip() == str(ref).strip()


def match_boolean(pred: Any, ref: Any) -> bool:
    """Match boolean answers. Exact match."""
    def to_bool(v):
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ("true", "1"):
            return True
        if s in ("false", "0"):
            return False
        return None

    return to_bool(pred) == to_bool(ref)


def match_list(pred: Any, ref: Any) -> bool:
    """Match list answers position by position."""
    if not isinstance(pred, list) or not isinstance(ref, list):
        return False
    if len(pred) != len(ref):
        return False
    for p, r in zip(pred, ref):
        if isinstance(r, (int, float)):
            if not match_number(p, r):
                return False
        else:
            if not match_string(p, r):
                return False
    return True


def match_table(pred: Any, ref: Any) -> bool:
    """Match table answers: columns by name, rows as sets."""
    if not isinstance(pred, dict) or not isinstance(ref, dict):
        return False

    pred_cols = pred.get("columns", [])
    ref_cols = ref.get("columns", [])
    pred_rows = pred.get("rows", [])
    ref_rows = ref.get("rows", [])

    # Column matching
    pred_cols_norm = [normalize_string(c) for c in pred_cols]
    ref_cols_norm = [normalize_string(c) for c in ref_cols]

    if pred_cols_norm == ref_cols_norm:
        col_order = list(range(len(pred_cols)))
    elif set(pred_cols_norm) == set(ref_cols_norm):
        # Reorder pred columns to match ref
        col_order = [pred_cols_norm.index(c) for c in ref_cols_norm]
    else:
        # Columns don't match by name, try positional
        if len(pred_cols) != len(ref_cols):
            return False
        col_order = list(range(len(pred_cols)))

    # Row matching (as sets)
    if len(pred_rows) != len(ref_rows):
        return False

    # Reorder pred row columns
    reordered_pred = []
    for row in pred_rows:
        reordered_pred.append(tuple(row[i] for i in col_order))

    ref_tuples = [tuple(row) for row in ref_rows]

    # Match rows as sets
    matched = [False] * len(ref_tuples)
    for pred_row in reordered_pred:
        found = False
        for j, ref_row in enumerate(ref_tuples):
            if matched[j]:
                continue
            if _rows_match(pred_row, ref_row):
                matched[j] = True
                found = True
                break
        if not found:
            return False

    return all(matched)


def _rows_match(pred_row: tuple, ref_row: tuple) -> bool:
    """Check if two table rows match cell by cell."""
    if len(pred_row) != len(ref_row):
        return False
    for p, r in zip(pred_row, ref_row):
        if isinstance(r, (int, float)):
            if not match_number(p, r):
                return False
        else:
            if not match_string(p, r):
                return False
    return True


# ---------------------------------------------------------------------------
# Main scoring
# ---------------------------------------------------------------------------

_MATCHERS = {
    "number": match_number,
    "percent": match_percent,
    "string": match_string,
    "date": match_date,
    "boolean": match_boolean,
    "list": match_list,
    "table": match_table,
}


def score_answer(pred: Dict, ref: Dict) -> int:
    """
    Score a single answer.

    Returns:
        +1 if correct
         0 if abstain
        -1 if wrong
    """
    pred_type = pred.get("type", "abstain")
    ref_type = ref.get("type")

    # Abstain (or missing answer)
    if pred_type == "abstain":
        return 0

    # Unanswerable
    if pred_type == "unanswerable":
        if ref_type == "unanswerable":
            return 1
        else:
            return -1

    # Type mismatch
    if pred_type != ref_type:
        return -1

    # For unanswerable reference, any non-abstain answer is wrong
    if ref_type == "unanswerable":
        return -1

    # Match values
    matcher = _MATCHERS.get(pred_type)
    if matcher is None:
        return -1

    pred_answer = pred.get("answer")
    ref_answer = ref.get("answer")
    aliases = ref.get("alias", [])

    if pred_type == "string":
        if matcher(pred_answer, ref_answer, aliases):
            return 1
    else:
        if matcher(pred_answer, ref_answer):
            return 1

    return -1


def compute_pa(predictions: Dict[str, Dict], references: Dict[str, Dict]) -> Dict:
    """
    Compute Penalized Accuracy (PA) over a set of questions.

    Parameters
    ----------
    predictions : Dict[str, Dict]
        Mapping from question_id to prediction dict.
    references : Dict[str, Dict]
        Mapping from question_id to reference dict.

    Returns
    -------
    Dict
        Results with PA score and detailed breakdown.
    """
    correct = 0
    wrong = 0
    abstain = 0
    missing = 0
    total = len(references)

    per_type_scores = {}
    details = []

    for q_id, ref in references.items():
        pred = predictions.get(q_id)

        if pred is None:
            # Missing = abstain
            score = 0
            missing += 1
        else:
            score = score_answer(pred, ref)

        ref_type = ref.get("type", "unknown")
        if ref_type not in per_type_scores:
            per_type_scores[ref_type] = {"correct": 0, "wrong": 0, "abstain": 0, "total": 0}

        per_type_scores[ref_type]["total"] += 1

        if score == 1:
            correct += 1
            per_type_scores[ref_type]["correct"] += 1
        elif score == -1:
            wrong += 1
            per_type_scores[ref_type]["wrong"] += 1
        else:
            abstain += 1
            per_type_scores[ref_type]["abstain"] += 1

        details.append({
            "id": q_id,
            "score": score,
            "pred_type": pred.get("type") if pred else "missing",
            "ref_type": ref_type,
        })

    pa = (correct - wrong) / total if total > 0 else 0.0

    return {
        "pa": round(pa, 4),
        "penalized_accuracy": round(pa, 4),
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "abstain": abstain,
        "missing": missing,
        "accuracy": round(correct / total, 4) if total > 0 else 0.0,
        "per_type": per_type_scores,
        "details": details,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DataMate Local Evaluator")
    parser.add_argument("--submission", type=str, required=True,
                        help="Path to submission.jsonl")
    parser.add_argument("--answers", type=str, required=True,
                        help="Path to answers.jsonl (reference)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to write detailed results JSON")
    parser.add_argument("--subset-only", action="store_true",
                        help="Evaluate only on questions present in the submission file")

    args = parser.parse_args()

    # Load predictions
    predictions = {}
    with open(args.submission, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                predictions[entry["id"]] = entry

    # Load references
    references = {}
    with open(args.answers, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                if not args.subset_only or entry["id"] in predictions:
                    references[entry["id"]] = entry

    # Compute PA
    results = compute_pa(predictions, references)

    # Print results
    print(f"\n{'='*50}")
    print(f"DataMate Evaluation Results")
    print(f"{'='*50}")
    print(f"  PA Score:  {results['pa']:.4f}")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    print(f"  Correct:   {results['correct']}/{results['total']}")
    print(f"  Wrong:     {results['wrong']}/{results['total']}")
    print(f"  Abstain:   {results['abstain']}/{results['total']}")
    print(f"  Missing:   {results['missing']}/{results['total']}")
    print()
    print("  Per-type breakdown:")
    for t, scores in results["per_type"].items():
        pa_t = (scores["correct"] - scores["wrong"]) / scores["total"] if scores["total"] > 0 else 0
        print(f"    {t:15s}: PA={pa_t:.3f}  "
              f"(✓{scores['correct']} ✗{scores['wrong']} ○{scores['abstain']} "
              f"/ {scores['total']})")
    print(f"{'='*50}\n")

    # Write detailed results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Detailed results written to {args.output}")


if __name__ == "__main__":
    main()
