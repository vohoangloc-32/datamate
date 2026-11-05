"""
confidence_gate.py — Confidence-Based Abstention for DataMate

Since PA penalizes wrong answers (-1) but abstain = 0, this module
implements heuristics to decide when to abstain rather than risk
a wrong answer.
"""

import logging
import math
from typing import Any, Dict, Optional
from .accounting_cleaner import normalize_text
from .question_understanding import QuestionInfo

logger = logging.getLogger(__name__)


class ConfidenceGate:
    """
    Decides whether to submit an answer or abstain based on
    various confidence signals.
    """

    def __init__(
        self,
        min_confidence: float = 0.45,
        enable_dual_run: bool = False,
    ):
        """
        Parameters
        ----------
        min_confidence : float
            Minimum confidence score to submit an answer (0-1).
        enable_dual_run : bool
            If True, support dual-run consistency checking.
        """
        self.min_confidence = min_confidence
        self.enable_dual_run = enable_dual_run

    def evaluate(
        self,
        answer: Dict[str, Any],
        question_info: Optional[QuestionInfo] = None,
        execution_retries: int = 0,
        max_retries: int = 3,
        second_answer: Optional[Dict[str, Any]] = None,
        question: str = "",
    ) -> Dict[str, Any]:
        """
        Evaluate whether to submit the answer or abstain.
        """
        # Already abstain or unanswerable — pass through
        if answer.get("type") in ("abstain", "unanswerable"):
            return answer

        confidence = 1.0
        reasons = []

        # 1. Penalty for retries (more retries = less confidence)
        if execution_retries > 0:
            retry_penalty = 0.35 * execution_retries
            confidence -= retry_penalty
            reasons.append(f"retries={execution_retries}")

        # 2. Question classification confidence
        if question_info and question_info.confidence < 0.4:
            confidence -= 0.2
            reasons.append(f"low_type_confidence={question_info.confidence:.2f}")

        # 3. Check for implausible values
        implausible = self._check_plausibility(answer)
        if implausible:
            confidence -= 0.35
            reasons.append(f"implausible: {implausible}")

        # 4. Zero numeric trap & high-risk percent checks
        answer_type = answer.get("type")
        val = answer.get("answer")
        if answer_type in ("number", "percent") and val is not None:
            try:
                num_val = float(val)
                if num_val == 0.0:
                    if execution_retries > 0:
                        confidence -= 0.35
                        reasons.append("zero_numeric_after_retry")
                    elif question_info and question_info.aggregation in ("sum", "mean"):
                        confidence -= 0.25
                        reasons.append("zero_numeric_on_aggregation")
                if answer_type == "percent":
                    if execution_retries >= 1:
                        confidence -= 0.35
                        reasons.append("percent_after_retry")
                    if num_val in (100.0, -100.0):
                        confidence -= 0.70
                        reasons.append("degenerate_percent_value")
                    elif num_val == 0.0 and execution_retries > 0:
                        confidence -= 0.35
                        reasons.append("zero_percent_after_retry")
            except (ValueError, TypeError):
                pass

        # 5. Empty collections check
        if answer_type == "list":
            if not isinstance(val, list) or len(val) == 0:
                confidence -= 0.6
                reasons.append("empty_or_invalid_list")
        elif answer_type == "string":
            if not val or not str(val).strip():
                confidence -= 0.6
                reasons.append("empty_string")
        elif answer_type == "table":
            if not isinstance(val, dict) or not val.get("rows"):
                confidence -= 0.6
                reasons.append("empty_table")

        # 6. Type mismatch between expected and actual (Critical: wrong type = -1 point)
        if question_info and question_info.expected_type and answer_type:
            if question_info.expected_type != answer_type:
                if question_info.confidence >= 0.6:
                    confidence -= 0.55
                    reasons.append(
                        f"type_mismatch: expected={question_info.expected_type}, "
                        f"got={answer_type}"
                    )
                else:
                    confidence -= 0.25
                    reasons.append(
                        f"weak_type_mismatch: expected={question_info.expected_type}, "
                        f"got={answer_type}"
                    )

        # 7. Dual-run consistency check
        if self.enable_dual_run and second_answer is not None:
            if not self._answers_match(answer, second_answer):
                confidence -= 0.5
                reasons.append("dual_run_inconsistent")

        # Decision
        if confidence < self.min_confidence:
            logger.info("Abstaining (confidence=%.2f, reasons=%s)", confidence, reasons)
            return {
                "type": "abstain",
                "reason": f"low_confidence={confidence:.2f}; " + "; ".join(reasons),
            }

        logger.debug("Accepting answer (confidence=%.2f)", confidence)
        return answer

    def _check_plausibility(self, answer: Dict[str, Any]) -> Optional[str]:
        """Check if the answer seems implausible."""
        answer_type = answer.get("type")
        value = answer.get("answer")

        if value is None:
            return None

        if answer_type == "number":
            try:
                num = float(value)
                if math.isnan(num) or math.isinf(num):
                    return "NaN or Inf"
                # Extremely large numbers might be suspect
                if abs(num) > 1e15:
                    return f"extremely_large={num}"
            except (ValueError, TypeError):
                return "not_a_number"

        elif answer_type == "percent":
            try:
                num = float(value)
                if math.isnan(num) or math.isinf(num):
                    return "NaN or Inf"
                # Percentages > 10000% or < -10000% are suspect
                if abs(num) > 10000:
                    return f"extreme_percent={num}"
            except (ValueError, TypeError):
                return "not_a_number"

        elif answer_type == "date":
            import re
            s = str(value)
            if not re.match(r"^\d{4}-\d{2}(-\d{2})?$", s):
                return f"invalid_date_format={s}"

        elif answer_type == "string":
            if len(str(value)) > 500:
                return "string_too_long"

        elif answer_type == "list":
            if isinstance(value, list) and len(value) == 0:
                return "empty_list"
            if isinstance(value, list) and len(value) > 100:
                return "list_too_long"

        return None

    def _answers_match(self, a1: Dict[str, Any], a2: Dict[str, Any]) -> bool:
        """Check if two answers are consistent."""
        if a1.get("type") != a2.get("type"):
            return False

        v1 = a1.get("answer")
        v2 = a2.get("answer")

        if v1 is None or v2 is None:
            return v1 == v2

        answer_type = a1.get("type")

        if answer_type in ("number", "percent"):
            try:
                return abs(float(v1) - float(v2)) < 0.01
            except (ValueError, TypeError):
                return str(v1) == str(v2)
        return str(v1) == str(v2)
