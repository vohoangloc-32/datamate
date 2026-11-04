"""Structured query planning for LLM-generated tabular answers."""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QueryPlan:
    """A constrained intermediate representation between a question and code."""

    metric: str = ""
    aggregation: str = ""
    entity_column: str = ""
    date_column: str = ""
    start_date: str = ""
    end_date: str = ""
    filters: List[str] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    source_tables: List[str] = field(default_factory=list)
    answer_type: str = ""
    confidence: float = 0.0
    assumptions: List[str] = field(default_factory=list)
    reason: str = ""
    order_id_column: str = ""
    measure_column: str = ""
    status_column: str = ""
    join_keys: List[str] = field(default_factory=list)
    distinct_required: bool = False
    period_inclusive: bool = True
    requirement_coverage: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "aggregation": self.aggregation,
            "entity_column": self.entity_column,
            "date_column": self.date_column,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "filters": self.filters,
            "group_by": self.group_by,
            "source_tables": self.source_tables,
            "answer_type": self.answer_type,
            "confidence": self.confidence,
            "assumptions": self.assumptions,
            "reason": self.reason,
            "order_id_column": self.order_id_column,
            "measure_column": self.measure_column,
            "status_column": self.status_column,
            "join_keys": self.join_keys,
            "distinct_required": self.distinct_required,
            "period_inclusive": self.period_inclusive,
            "requirement_coverage": self.requirement_coverage,
        }


def _parse_object(response: str) -> Dict[str, Any]:
    raw = response.strip()
    if "```" in raw:
        raw = raw.replace("```json", "").replace("```", "").strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError("query plan response did not contain a JSON object")
    value = json.loads(raw[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("query plan must be a JSON object")
    return value


def build_query_plan(
    verifier_fn: Optional[Callable[[str], str]],
    *,
    question: str,
    as_of_date: str,
    schema: str,
    rulebook: str,
    tables: Dict[str, pd.DataFrame],
    table_var_names: Dict[str, str],
    expected_type: str,
) -> Dict[str, Any]:
    """Ask the model for a plan and reject plans that reference unavailable data."""
    fallback = {
        "status": "fallback",
        "metric": "",
        "aggregation": "",
        "entity_column": "",
        "date_column": "",
        "start_date": "",
        "end_date": "",
        "filters": [],
        "group_by": [],
        "source_tables": [],
        "answer_type": expected_type,
        "confidence": 0.0,
        "assumptions": [],
        "reason": "query_plan_verifier_unavailable",
        "order_id_column": "",
        "measure_column": "",
        "status_column": "",
        "join_keys": [],
        "distinct_required": False,
        "period_inclusive": True,
        "requirement_coverage": {},
    }
    if verifier_fn is None:
        return fallback

    prompt = (
        "You are a structured query planner for a Vietnamese tabular QA agent. "
        "Do not write Python. Resolve the question into a plan using only the "
        "listed tables and columns. A question asking for order count means "
        "distinct order IDs, not detail rows. A historical period is answerable "
        "when matching data tables exist, even if it differs from as_of_date. "
        "Return ONLY JSON with exactly these fields: metric, aggregation, "
        "entity_column, date_column, start_date, end_date, filters (string array), "
        "group_by (string array), source_tables (string array), answer_type, "
        "confidence (0..1), assumptions (string array), reason, "
        "order_id_column, measure_column, status_column, join_keys (string array), "
        "distinct_required (boolean), period_inclusive (boolean), "
        "requirement_coverage (object mapping requirement to operation).\n\n"
        f"Question: {question}\nAs-of date: {as_of_date}\n"
        f"Expected answer type: {expected_type}\n"
        f"Available table variables: {json.dumps(table_var_names, ensure_ascii=False)}\n"
        f"Available schema:\n{schema}\nBusiness rules:\n{rulebook}"
    )
    try:
        value = _parse_object(verifier_fn(prompt))
        plan = dict(fallback)
        for key in plan:
            if key in value:
                plan[key] = value[key]
        if "distinct_required" not in value and plan["aggregation"] == "count_distinct":
            plan["distinct_required"] = True
        if "order_id_column" not in value and plan["aggregation"] == "count_distinct":
            plan["order_id_column"] = plan["entity_column"]
        if not isinstance(plan["confidence"], (int, float)) or not 0 <= float(plan["confidence"]) <= 1:
            raise ValueError("query plan confidence must be between 0 and 1")
        for key in ("filters", "group_by", "source_tables", "assumptions", "join_keys"):
            if not isinstance(plan[key], list) or not all(isinstance(item, str) for item in plan[key]):
                raise ValueError(f"{key} must be a string array")
        if not isinstance(plan["requirement_coverage"], dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in plan["requirement_coverage"].items()
        ):
            raise ValueError("requirement_coverage must be a string mapping")
        for key in ("distinct_required", "period_inclusive"):
            if not isinstance(plan[key], bool):
                raise ValueError(f"{key} must be boolean")
        available_tables = set(tables)
        if plan["source_tables"] and not set(plan["source_tables"]).issubset(available_tables):
            raise ValueError("query plan references unavailable tables")
        available_columns = {
            str(column) for frame in tables.values() for column in frame.columns
        }
        allowed_answer_types = {
            "number", "percent", "string", "date", "boolean",
            "list", "table", "unanswerable",
        }
        if plan["answer_type"] not in allowed_answer_types:
            raise ValueError(f"invalid answer_type: {plan['answer_type']}")
        allowed_aggregations = {
            "", "sum", "avg", "count", "count_distinct",
            "min", "max", "ratio", "difference",
        }
        if plan["aggregation"] not in allowed_aggregations:
            raise ValueError(f"invalid aggregation: {plan['aggregation']}")
        if not plan["source_tables"] and plan["answer_type"] != "unanswerable":
            raise ValueError("query plan must identify at least one source table")
        for key in (
            "entity_column", "date_column", "order_id_column",
            "measure_column", "status_column",
        ):
            if plan[key] and plan[key] not in available_columns:
                raise ValueError(f"query plan references unavailable column: {plan[key]}")
        for key in ("metric", "aggregation", "answer_type", "start_date", "end_date"):
            if not isinstance(plan[key], str):
                raise ValueError(f"{key} must be a string")
        question_key = question.casefold()
        order_question = any(term in question_key for term in ("đơn", "hóa đơn", "hoá đơn", "bill", "order"))
        revenue_question = any(term in question_key for term in ("doanh thu", "tiền bán", "tiền hàng", "doanh số"))
        if order_question and plan["aggregation"] in {"count", "count_distinct"}:
            if plan["aggregation"] != "count_distinct" or not plan["distinct_required"]:
                raise ValueError("order counts must use distinct order IDs")
            if not plan["entity_column"] and not plan["order_id_column"]:
                raise ValueError("order counts require an order ID column")
        if revenue_question and plan["aggregation"] not in {"", "sum"}:
            raise ValueError("revenue queries must use sum aggregation")
        plan["status"] = "validated"
        return plan
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Query plan response ignored: %s", exc)
        fallback["reason"] = f"query_plan_invalid: {exc}"
        return fallback
