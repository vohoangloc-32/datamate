"""
confidence_gate.py — Confidence-Based Abstention for DataMate

Since PA penalizes wrong answers (-1) but abstain = 0, this module
implements heuristics to decide when to abstain rather than risk
a wrong answer.
"""

import json
import logging
import math
from datetime import date
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .question_understanding import QuestionInfo

logger = logging.getLogger(__name__)


class ConfidenceGate:
    """
    Decides whether to submit an answer or abstain based on
    various confidence signals.
    """

    def __init__(
        self,
        min_confidence: float = 0.3,
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
    ) -> Dict[str, Any]:
        """
        Evaluate whether to submit the answer or abstain.

        Parameters
        ----------
        answer : Dict[str, Any]
            The formatted answer dict (with "type" and "answer" keys).
        question_info : QuestionInfo, optional
            Analyzed question information.
        execution_retries : int
            Number of retries that were needed.
        max_retries : int
            Maximum retries allowed.
        second_answer : Dict[str, Any], optional
            A second answer from a dual-run (for consistency check).

        Returns
        -------
        Dict[str, Any]
            Either the original answer or an abstain entry.
        """
        # Already abstain or unanswerable — pass through
        if answer.get("type") in ("abstain", "unanswerable"):
            return answer

        confidence = 1.0
        reasons = []

        # 1. Penalty for retries (more retries = less confidence)
        if execution_retries > 0:
            retry_penalty = 0.07 * execution_retries
            confidence -= retry_penalty
            reasons.append(f"retries={execution_retries}")

        # 2. Question classification confidence
        if question_info and question_info.confidence < 0.4:
            confidence -= 0.07
            reasons.append(f"low_type_confidence={question_info.confidence:.2f}")

        # 3. Check for implausible values
        implausible = self._check_plausibility(answer)
        if implausible:
            confidence -= 0.13
            reasons.append(f"implausible: {implausible}")

        # 4. Dual-run consistency check
        if self.enable_dual_run and second_answer is not None:
            if not self._answers_match(answer, second_answer):
                confidence -= 0.2
                reasons.append("dual_run_inconsistent")

        # 5. Type mismatch between expected and actual
        if question_info:
            if (question_info.expected_type != answer.get("type") and
                question_info.confidence > 0.5):
                confidence -= 0.07
                reasons.append(
                    f"type_mismatch: expected={question_info.expected_type}, "
                    f"got={answer.get('type')}"
                )

        # Decision
        if confidence < self.min_confidence:
            logger.info("Abstaining (confidence=%.2f, reasons=%s)", confidence, reasons)
            return {
                "type": "abstain",
                "reason": f"low_confidence={confidence:.2f}; " + "; ".join(reasons),
            }

        logger.debug("Accepting answer (confidence=%.2f)", confidence)
        return answer

    def evaluate_runs(
        self,
        answers: List[Dict[str, Any]],
        *,
        candidate_records: Optional[List[Dict[str, Any]]] = None,
        failed_runs: int = 0,
        execution_retries: int = 0,
        question_info: Optional[QuestionInfo] = None,
        prompt_verifier_fn: Optional[Callable[[str], str]] = None,
        verifier_context: Optional[Dict[str, Any]] = None,
        completeness_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Rank verified candidates, then fall back to modal-answer scoring."""
        if not answers:
            return {"type": "abstain", "reason": "no_runs"}

        valid_answers = [
            answer for answer in answers
            if answer.get("type") not in ("abstain",)
        ]
        if not valid_answers:
            return {
                "type": "abstain",
                "reason": f"all_runs_failed={failed_runs}",
            }

        if candidate_records and prompt_verifier_fn is not None:
            ranked = self._rank_candidates(
                candidate_records,
                prompt_verifier_fn=prompt_verifier_fn,
                verifier_context=verifier_context or {},
                question_info=question_info,
            )
            if ranked is not None:
                selected, score, reason = ranked
                if score >= self.min_confidence:
                    return selected
                return {
                    "type": "abstain",
                    "reason": f"low_confidence={score:.2f}; {reason}",
                }

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for answer in valid_answers:
            key = self._answer_key(answer)
            groups.setdefault(key, []).append(answer)
        winning_key, winning_group = max(
            groups.items(),
            key=lambda item: (len(item[1]), item[0]),
        )
        selected = dict(winning_group[0])

        confidence = 1.0
        reasons: List[str] = []
        run_count = len(answers)
        if completeness_context:
            completeness_status = completeness_context.get("question_status")
            completeness_confidence = completeness_context.get("confidence")
            if isinstance(completeness_confidence, (int, float)):
                confidence -= max(0.0, 1.0 - float(completeness_confidence)) * 0.07
                reasons.append(
                    f"question_completeness_confidence={float(completeness_confidence):.2f}"
                )
            if completeness_status == "ambiguous":
                confidence -= 0.05
                reasons.append("question_ambiguous")
            elif completeness_context.get("missing_context"):
                confidence -= 0.03
                reasons.append("question_missing_context")
        duplicate_count = len(winning_group)
        if duplicate_count > 1:
            duplicate_bonus = min(0.2, 0.05 * (duplicate_count - 1))
            confidence += duplicate_bonus
            reasons.append(f"matching_runs={duplicate_count}")
        elif len(groups) > 1:
            confidence = 0.0
            reasons.append("no_consensus")

        if len(groups) > 1:
            type_count = len({answer.get("type") for answer in valid_answers})
            if type_count > 1:
                confidence -= 0.33
                reasons.append(f"answer_type_variance={type_count}")
            else:
                reasons.append(f"value_variance={len(groups)}")

        if failed_runs:
            confidence -= min(0.2, 0.07 * failed_runs)
            reasons.append(f"failed_runs={failed_runs}/{run_count}")

        if execution_retries:
            confidence -= min(0.2, 0.03 * execution_retries)
            reasons.append(f"retries={execution_retries}")

        if question_info:
            if question_info.confidence < 0.4:
                confidence -= 0.07
                reasons.append(f"low_type_confidence={question_info.confidence:.2f}")
            if (
                question_info.expected_type != selected.get("type")
                and question_info.confidence > 0.5
            ):
                confidence -= 0.07
                reasons.append(
                    f"type_mismatch: expected={question_info.expected_type}, "
                    f"got={selected.get('type')}"
                )

        implausible = self._check_plausibility(selected)
        if implausible:
            confidence -= 0.07
            reasons.append(f"implausible: {implausible}")

        if prompt_verifier_fn is not None:
            verdict = self._run_prompt_verifier(
                prompt_verifier_fn,
                answers=answers,
                selected=selected,
                context=verifier_context or {},
            )
            if verdict is not None:
                verifier_confidence = verdict["confidence"]
                confidence = min(confidence, verifier_confidence + 0.1)
                reasons.append(
                    f"prompt_verifier_confidence={verifier_confidence:.2f}"
                )
                if not verdict["is_consistent"]:
                    confidence -= 0.13
                    reasons.append("prompt_verifier_inconsistent")

        confidence = max(0.0, min(1.0, confidence))
        if confidence < self.min_confidence:
            logger.info(
                "Abstaining after repeated runs (confidence=%.2f, reasons=%s)",
                confidence,
                reasons,
            )
            return {
                "type": "abstain",
                "reason": f"low_confidence={confidence:.2f}; " + "; ".join(reasons),
            }
        return selected

    def _rank_candidates(
        self,
        candidate_records: List[Dict[str, Any]],
        *,
        prompt_verifier_fn: Callable[[str], str],
        verifier_context: Dict[str, Any],
        question_info: Optional[QuestionInfo],
    ) -> Optional[tuple[Dict[str, Any], float, str]]:
        """Rank executable candidates with per-candidate semantic verification."""
        ranked: List[tuple[float, Dict[str, Any], str]] = []
        answer_counts: Dict[str, int] = {}
        for record in candidate_records:
            answer = record.get("answer")
            if isinstance(answer, dict) and answer.get("type") not in ("abstain", "unanswerable"):
                key = self._answer_key(answer)
                answer_counts[key] = answer_counts.get(key, 0) + 1

        for record in candidate_records:
            answer = record.get("answer")
            if not isinstance(answer, dict) or answer.get("type") in ("abstain", "unanswerable"):
                continue
            verdict = self._run_candidate_verifier(
                prompt_verifier_fn,
                answer=answer,
                record=record,
                context=verifier_context,
            )
            if verdict is None:
                continue
            score = 0.65 * verdict["confidence"]
            reasons = [f"candidate_verifier_confidence={verdict['confidence']:.2f}"]
            if verdict["confidence"] < 0.1:
                ranked.append((0.0, answer, "candidate_verifier_confidence_below_floor"))
                continue
            if verdict["is_consistent"]:
                score += 0.15
            else:
                score -= 0.25
                reasons.append("candidate_verifier_inconsistent")
            if record.get("execution_success", True):
                score += 0.1
            else:
                score -= 0.3
                reasons.append("execution_failed")
            if question_info and question_info.expected_type == answer.get("type"):
                score += 0.1
            elif question_info:
                score -= 0.15
                reasons.append("type_mismatch")
            answer_count = answer_counts.get(self._answer_key(answer), 1)
            consensus_bonus = min(0.1, 0.05 * (answer_count - 1))
            score += consensus_bonus
            if consensus_bonus:
                reasons.append(f"matching_candidates={answer_count}")
            retries = int(record.get("retries", 0) or 0)
            score -= min(0.12, 0.03 * retries)
            if retries:
                reasons.append(f"retries={retries}")
            implausible = self._check_plausibility(answer)
            if implausible:
                score -= 0.15
                reasons.append(f"implausible={implausible}")
            ranked.append((max(0.0, min(1.0, score)), answer, "; ".join(reasons)))
        if not ranked:
            return None
        score, answer, reason = max(ranked, key=lambda item: item[0])
        return dict(answer), score, reason

    def evaluate_question_completeness(
        self,
        verifier_fn: Optional[Callable[[str], str]],
        *,
        question: str,
        as_of_date: str,
        question_info: QuestionInfo,
        schema: str,
        rulebook: str,
        business_context: str = "",
    ) -> Dict[str, Any]:
        """Check whether the question has enough context before code generation."""
        fallback = {
            "question_status": "answerable",
            "confidence": question_info.confidence,
            "missing_context": [],
            "assumptions": [],
            "reason": "pattern_analysis_fallback",
        }
        if verifier_fn is None:
            return fallback

        prompt = (
            "You are a strict question-completeness verifier for a data QA agent. "
            "Decompose the question into metric, time range, filters, grouping, "
            "ranking, and requested output fields. Use the business context, schema, "
            "as-of date, and rules. Missing context is acceptable only when a clear "
            "rule supplies a safe default. Return ONLY JSON with keys: "
            '"question_status" (one of "answerable", "ambiguous", "unanswerable"), '
            '"confidence" (number 0..1), "missing_context" (array of strings), '
            '"assumptions" (array of strings), and "reason" (short string). '
            "Use unanswerable when different plausible interpretations can change "
            "the answer and no rule resolves them.\n\n"
            f"Question: {question}\n"
            f"As-of date: {as_of_date}\n"
            f"Pattern analysis: {json.dumps(question_info.to_dict(), ensure_ascii=False)}\n"
            f"Business context:\n{business_context}\n"
            f"Schema:\n{schema}\n"
            f"Business rules:\n{rulebook}"
        )
        try:
            verdict = self._parse_verifier_response(verifier_fn(prompt))
            status = verdict.get("question_status")
            confidence = float(verdict.get("confidence"))
            if status not in {"answerable", "ambiguous", "unanswerable"}:
                raise ValueError("invalid question_status")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between 0 and 1")
            missing = verdict.get("missing_context", [])
            assumptions = verdict.get("assumptions", [])
            if not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
                raise ValueError("missing_context must be a string array")
            if not isinstance(assumptions, list) or not all(isinstance(item, str) for item in assumptions):
                raise ValueError("assumptions must be a string array")
            return {
                "question_status": status,
                "confidence": confidence,
                "missing_context": missing,
                "assumptions": assumptions,
                "reason": str(verdict.get("reason", "")),
            }
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
            logger.warning("Question completeness response ignored: %s", exc)
            return fallback

    @staticmethod
    def _run_prompt_verifier(
        verifier_fn: Callable[[str], str],
        *,
        answers: List[Dict[str, Any]],
        selected: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Ask the model to check semantic agreement without trusting free text."""
        prompt = (
            "You are a strict answer verifier. Compare the candidate answers below "
            "using the question, expected type, schema, and business rules. Do not "
            "write code or recalculate from unsupported assumptions. Return ONLY JSON "
            'with keys "is_consistent" (boolean), "confidence" (number 0..1), '
            '"reason" (short string). Mark is_consistent false when there is no '
            "defensible consensus.\n\n"
            f"Question: {context.get('question', '')}\n"
            f"As-of date: {context.get('as_of_date', '')}\n"
            f"Expected answer type: {context.get('expected_type', '')}\n"
            f"Schema:\n{context.get('schema', '')}\n"
            f"Business rules:\n{context.get('rulebook', '')}\n"
            f"Selected majority answer:\n{json.dumps(selected, ensure_ascii=False, default=str)}\n"
            f"All candidate answers:\n{json.dumps(answers, ensure_ascii=False, default=str)}"
        )
        try:
            verdict = ConfidenceGate._parse_verifier_response(verifier_fn(prompt))
            if not isinstance(verdict, dict):
                raise ValueError("verifier response is not an object")
            is_consistent = verdict.get("is_consistent")
            confidence = verdict.get("confidence")
            if not isinstance(is_consistent, bool):
                raise ValueError("is_consistent must be boolean")
            confidence = float(confidence)
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between 0 and 1")
            return {
                "is_consistent": is_consistent,
                "confidence": confidence,
            }
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
            logger.warning("Prompt verifier response ignored: %s", exc)
            return None

    @staticmethod
    def _run_candidate_verifier(
        verifier_fn: Callable[[str], str],
        *,
        answer: Dict[str, Any],
        record: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Verify one candidate's semantic correctness and execution evidence."""
        prompt = (
            "You are a strict candidate answer verifier. Decide whether this "
            "single candidate correctly answers the question using the schema "
            "and business rules. Consider the execution output and code evidence. "
            "Return ONLY JSON with keys "
            '"is_consistent" (boolean), "confidence" (number 0..1), '
            '"reason" (short string).\n\n'
            f"Question: {context.get('question', '')}\n"
            f"As-of date: {context.get('as_of_date', '')}\n"
            f"Expected answer type: {context.get('expected_type', '')}\n"
            f"Schema:\n{context.get('schema', '')}\n"
            f"Business rules:\n{context.get('rulebook', '')}\n"
            f"Candidate answer:\n{json.dumps(answer, ensure_ascii=False, default=str)}\n"
            f"Execution success: {record.get('execution_success', True)}\n"
            f"Execution output:\n{record.get('execution_output', '')}\n"
            f"Generated code:\n{record.get('code', '')}"
        )
        try:
            verdict = ConfidenceGate._parse_verifier_response(verifier_fn(prompt))
            confidence = float(verdict.get("confidence"))
            if not isinstance(verdict.get("is_consistent"), bool):
                raise ValueError("is_consistent must be boolean")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence must be between 0 and 1")
            return {
                "is_consistent": verdict["is_consistent"],
                "confidence": confidence,
            }
        except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
            logger.warning("Candidate verifier response ignored: %s", exc)
            return None

    @staticmethod
    def _parse_verifier_response(response: str) -> Dict[str, Any]:
        raw = response.strip()
        if raw.startswith("```"):
            raw = raw.strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].lstrip()
        verdict = json.loads(raw)
        if not isinstance(verdict, dict):
            raise ValueError("verifier response is not an object")
        return verdict

    @staticmethod
    def _answer_key(answer: Dict[str, Any]) -> str:
        """Create a stable key for comparing JSON-compatible answers."""
        return json.dumps(
            {"type": answer.get("type"), "answer": answer.get("answer")},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

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
