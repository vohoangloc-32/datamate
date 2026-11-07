"""
confidence_gate.py — Confidence-Based Abstention for DataMate

Since PA penalizes wrong answers (-1) but abstain = 0, this module
implements heuristics to decide when to abstain rather than risk
a wrong answer.
"""

import ast
import logging
import math
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

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
        code: str = "",
        tables: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate whether to submit the answer or abstain using a 3-layer audit:
        1. Schema Answerability Audit (detect unrecorded entities like gender, cancel reason).
        2. Static AST Code Audit (missing temporal filter, unauthorized unit division, count vs sum).
        3. Domain Plausibility Gate & Calibrated Confidence Scoring.
        """
        # --- Layer 1: Schema Answerability Audit ---
        if question and tables:
            unanswerable_reason = self._audit_schema_answerability(question, tables)
            if unanswerable_reason:
                logger.info("Schema Answerability Gate: %s", unanswerable_reason)
                return {
                    "type": "unanswerable",
                    "reason": unanswerable_reason,
                }

        # Already abstain — pass through
        if answer.get("type") == "abstain":
            return answer

        # Guard against hallucinated unanswerable fallback from LLM (avoids -1 penalty)
        if answer.get("type") == "unanswerable":
            return {
                "type": "abstain",
                "reason": "unverified_unanswerable_converted_to_abstain",
            }

        confidence = 1.0
        reasons = []

        # --- Layer 2: Static AST Code Audit ---
        if code:
            ast_penalties = self._audit_code_ast(code, question, question_info)
            for penalty, reason in ast_penalties:
                confidence -= penalty
                reasons.append(reason)

        # --- Layer 3: Domain Plausibility Gate ---
        domain_penalties = self._check_domain_plausibility(answer, question, question_info)
        for penalty, reason in domain_penalties:
            confidence -= penalty
            reasons.append(reason)

        # 1. Calibrated penalty for retries (each retry indicates flaky code)
        if execution_retries > 0:
            retry_penalty = 0.25 * execution_retries
            confidence -= retry_penalty
            reasons.append(f"retries={execution_retries}")

        # 2. Question classification confidence
        if question_info and question_info.confidence < 0.4:
            confidence -= 0.2
            reasons.append(f"low_type_confidence={question_info.confidence:.2f}")

        # 3. Check for general implausible values (NaN, Inf, extreme size)
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
                        confidence -= 0.40
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
        elif answer_type == "list" and isinstance(val, list) and len(val) >= 2:
            try:
                num_list = [float(x) for x in val]
                if 0.0 in num_list and any(x > 10_000_000 for x in num_list):
                    confidence -= 0.60
                    reasons.append("list_contains_zero_with_large_revenues")
            except Exception:
                pass

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

    def _audit_schema_answerability(
        self,
        question: str,
        tables: Dict[str, pd.DataFrame],
    ) -> Optional[str]:
        """Detect questions asking for entities not recorded anywhere in the business schema."""
        q_norm = normalize_text(question)
        all_cols: Set[str] = set()
        table_names: Set[str] = set()
        for tname, df in tables.items():
            table_names.add(normalize_text(tname))
            for col in df.columns:
                if not str(col).startswith("__"):
                    all_cols.add(normalize_text(col))

        # 1. Customer gender
        if any(term in q_norm for term in ("gioi tinh", "khach nam", "khach nu", "khach hang nam", "khach hang nu")):
            if not any(any(k in col for k in ("gioi tinh", "gender", "sex", "phai")) for col in all_cols):
                return "customer_gender_not_in_schema"

        # 2. Specific cancel/return reasons
        if any(term in q_norm for term in ("ly do huy", "li do huy", "vi sao huy", "tai sao huy", "nguyen nhan huy", "vi giao tre", "vi het hang", "ly do tra", "nguyen nhan tra")):
            found_reason = False
            for df in tables.values():
                for col in df.columns:
                    if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                        sample_s = df[col].dropna().astype(str).str.lower()
                        if sample_s.str.contains(r"giao tre|het hang|doi y|sai hang", regex=True).any():
                            found_reason = True
                            break
                if found_reason:
                    break
            if not found_reason:
                return "cancel_reason_not_recorded_in_data"

        # 3. Customer age / birth year
        if any(term in q_norm for term in ("do tuoi", "bao nhieu tuoi", "nam sinh", "ngay sinh")):
            if not any(any(k in col for k in ("tuoi", "age", "nam sinh", "ngay sinh", "birth")) for col in all_cols):
                return "customer_age_not_in_schema"

        # 4. Import / Purchasing when no import table exists
        if any(term in q_norm for term in ("nhap hang", "nhap kho", "tien nhap", "phieu nhap", "chi phi nhap")):
            if not any("nhap" in t or "import" in t or "purchase" in t for t in table_names):
                return "import_table_not_in_schema"

        # 5. Profit / Margin when no cost column exists
        if any(term in q_norm for term in ("loi nhuan", "tien lai", "ty suat loi nhuan", "ti suat loi nhuan")):
            if not any(any(k in col for k in ("gia von", "giavon", "cost", "von", "loi nhuan")) for col in all_cols):
                return "cost_data_not_in_schema"

        return None

    def _audit_code_ast(
        self,
        code: str,
        question: str,
        question_info: Optional[QuestionInfo],
    ) -> List[Tuple[float, str]]:
        """Inspect generated code AST for logic mismatches, scale errors, and missing temporal filters."""
        penalties: List[Tuple[float, str]] = []
        q_norm = normalize_text(question)

        # 1. Missing temporal filter on time-constrained question
        has_time_constraint = False
        if question_info:
            if getattr(question_info, "time_expressions", None):
                has_time_constraint = len(question_info.time_expressions) > 0
            elif getattr(question_info, "time_ranges", None):
                has_time_constraint = any(
                    isinstance(tr, dict) and (tr.get("start") or tr.get("end"))
                    for tr in question_info.time_ranges
                )
        if not has_time_constraint:
            has_time_constraint = bool(
                re.search(r"\b(?:thang \d+|nam \d+|quy [1-4]|ngay \d+|hom nay|hom qua|tuan nay|tuan truoc)\b", q_norm)
            )

        if has_time_constraint:
            code_lower = code.lower()
            # Snapshot tables (e.g. inventory report) or monthly specific tables don't need explicit date filters
            is_snapshot_table = any(t in code_lower for t in ("tonkho", "baocaotonkho", "inventory", "stock"))
            is_month_specific_table = bool(re.search(r"__(?:t\d{1,2}_202\d|source_month)", code_lower))
            if not is_snapshot_table and not is_month_specific_table:
                time_tokens = (
                    "ngay", "date", "time", "thang", "nam", "quy",
                    "month", "year", "quarter", "day",
                    "timestamp", "to_datetime", "dt.", "between", "timedel"
                )
                has_time_token = any(tok in code_lower for tok in time_tokens) or bool(
                    re.search(r"['\"](?:202[4-6](?:-\d{2})?)['\"]", code)
                )
                if not has_time_token:
                    penalties.append((0.60, "missing_temporal_filter_in_code"))

        # 2. Unauthorized unit division (e.g. dividing by 1e9 / 1e6 when question did not ask for billions/millions)
        if re.search(r"/\s*(?:1(?:_?000){2,3}|1e[69]|10\s*\*\*\s*[69])\b", code):
            if not any(unit in q_norm for unit in ("ty", "trieu", "tr", "ty dong", "trieu dong")):
                penalties.append((0.70, "unauthorized_unit_division_in_code"))

        # 3. Count vs Sum mismatch: question asks for money but code uses row count / shape[0] without sum
        is_monetary = any(
            m in q_norm for m in ("doanh thu", "bao nhieu tien", "tien hang", "tong tien", "chi phi", "phai thu", "thanh toan")
        ) and not any(k in q_norm for k in ("ti le", "ty le", "%", "so don", "may don", "don hang"))
        if is_monetary:
            if re.search(r"result\s*=\s*(?:int\()?len\(|result\s*=\s*\w+\.shape\[0\]", code) and ".sum()" not in code:
                penalties.append((0.60, "count_used_for_monetary_question"))

        # 4. Single-day relative questions missing lower bound or using wrong month-long range
        is_single_day = any(d in q_norm for d in ("hom nay", "hnay", "bua nay", "hom qua", "hqua", "bua qua"))
        is_range_or_inventory = any(k in q_norm for k in ("tu ", "den ", "khoang", "giai doan", "trong ")) or any(t in q_norm for t in ("ton", "kho", "con lai", "hang ton"))
        if is_single_day and not is_range_or_inventory:
            # Check for two timestamps 1 day apart (e.g. >= pd.Timestamp('2026-05-02') & < pd.Timestamp('2026-05-03'))
            m_ts = re.findall(r"pd\.Timestamp\(['\"](\d{4}-\d{2}-\d{2})['\"]\)", code)
            ts_1day_apart = False
            if len(m_ts) >= 2:
                try:
                    from datetime import date as _dt_date
                    d0 = _dt_date.fromisoformat(m_ts[0])
                    d1 = _dt_date.fromisoformat(m_ts[1])
                    if abs((d1 - d0).days) == 1:
                        ts_1day_apart = True
                except Exception:
                    pass
            has_day_bound = (
                bool(re.search(r"\.dt\.(?:date|normalize\(\))\s*==|Timedelta\(\s*days\s*=\s*1\s*\)|Timedelta\(\s*['\"]1\s*d", code))
                or ts_1day_apart
                or bool(re.search(r"<=\s*previous_day\s*\+\s*pd\.Timedelta", code))
            )
            if not has_day_bound:
                penalties.append((0.70, "single_day_question_missing_day_boundary"))
            elif re.search(r"pd\.Timestamp\(['\"]202\d-\d{2}-01['\"]\)\s*.*pd\.Timestamp\(['\"]202\d-\d{2}-(?:28|29|30|31)['\"]\)", code, re.DOTALL):
                penalties.append((0.70, "single_day_question_uses_month_range"))

        return penalties

    def _check_domain_plausibility(
        self,
        answer: Dict[str, Any],
        question: str,
        question_info: Optional[QuestionInfo],
    ) -> List[Tuple[float, str]]:
        """Validate result against retail SME domain boundaries."""
        penalties: List[Tuple[float, str]] = []
        q_norm = normalize_text(question)
        val = answer.get("answer")
        ans_type = answer.get("type")

        if val is None or not isinstance(val, (int, float)):
            return penalties

        num = float(val)

        # 1. Store revenue in VND cannot be < 500 VND (hallucinated division by 1e9 or 1e6)
        is_revenue_vnd = (
            ans_type == "number"
            and any(m in q_norm for m in ("doanh thu", "tien hang", "tong tien", "ban duoc bao nhieu tien"))
            and not any(k in q_norm for k in (
                "ti le", "ty le", "%", "phan tram", "tang truong", "thay doi",
                "don hang", "so don", "don", "so luong", "ly", "chai", "hop", "cai", "sp", "san pham"
            ))
        )
        if is_revenue_vnd and 0.0 < num < 500.0:
            penalties.append((0.80, f"implausible_vnd_revenue={num}"))

        # 2. Implausible single-store order count > 500,000 orders
        is_order_count = any(
            m in q_norm for m in ("so don", "may don", "bao nhieu don", "don hang")
        ) and not any(k in q_norm for k in ("ti le", "ty le", "%", "doanh thu", "tien"))
        if is_order_count and num > 500_000:
            penalties.append((0.70, f"implausible_order_count={num}"))

        # 3. Order / bill count question returning currency-scale number (> 100,000)
        is_bill_count = any(b in q_norm for b in ("so bill", "soos bill", "may bill", "bao nhieu bill", "may don", "bao nhieu don", "may dot", "so dot"))
        if is_bill_count and ans_type == "number" and num > 100_000:
            penalties.append((0.80, f"implausible_bill_count={num}"))

        # 4. Percent boundaries: bounded shares [0, 100]% vs general percentage
        if ans_type == "percent":
            if abs(num) > 1000.0:
                penalties.append((0.70, f"extreme_percent={num}"))
            elif any(r in q_norm for r in ("ti le", "ty le", "phan tram", "chiem", "dong gop")) and not any(
                g in q_norm for g in ("tang truong", "thay doi", "tang", "giam")
            ):
                if num < 0.0 or num > 100.0:
                    penalties.append((0.70, f"rate_out_of_bounds_0_100={num}"))

        return penalties

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
