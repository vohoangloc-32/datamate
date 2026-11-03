"""Typed analysis plans and deterministic pandas execution."""

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .code_executor import ExecutionResult


class AnalysisPlanError(ValueError):
    """Raised when an analysis plan is invalid or unsupported."""


_PLAN_FIELDS = {
    "answer_type",
    "table",
    "aggregation",
    "value_column",
    "date_column",
    "start_date",
    "end_date_exclusive",
    "filters",
    "group_by",
    "output",
    "limit",
    "sort_by",
    "sort_direction",
}
_AGGREGATIONS = {"sum", "count", "nunique", "mean", "min", "max"}
_FILTER_OPERATORS = {"eq", "ne", "in", "not_in", "is_null", "not_null"}
_OUTPUTS = {"scalar", "group_keys", "group_values", "table"}
_ANSWER_TYPES = {"number", "string", "list", "table"}


@dataclass(frozen=True)
class AnalysisPlan:
    answer_type: str
    table: str
    aggregation: str
    value_column: Optional[str]
    date_column: Optional[str]
    start_date: Optional[str]
    end_date_exclusive: Optional[str]
    filters: Tuple[Dict[str, Any], ...]
    group_by: Tuple[str, ...]
    output: str
    limit: Optional[int]
    sort_by: str
    sort_direction: str

    @classmethod
    def from_response(
        cls,
        response: str,
        tables: Dict[str, pd.DataFrame],
        expected_type: str,
    ) -> "AnalysisPlan":
        text = response.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AnalysisPlanError(f"Plan is not valid JSON: {exc.msg}") from exc
        _ALLOWED_COMMENT_FIELDS = {"reason", "thought", "notes", "explanation", "comment"}
        _OPTIONAL_FIELDS = {
            "date_column",
            "start_date",
            "end_date_exclusive",
            "filters",
            "group_by",
            "limit",
            "sort_by",
            "sort_direction",
        }
        unknown = set(payload) - _PLAN_FIELDS - _ALLOWED_COMMENT_FIELDS
        missing = (_PLAN_FIELDS - _OPTIONAL_FIELDS) - set(payload)
        if unknown:
            raise AnalysisPlanError(f"Unknown plan fields: {sorted(unknown)}")
        if missing:
            raise AnalysisPlanError(f"Missing plan fields: {sorted(missing)}")

        payload.setdefault("date_column", None)
        payload.setdefault("start_date", None)
        payload.setdefault("end_date_exclusive", None)
        payload.setdefault("filters", [])
        payload.setdefault("group_by", [])
        payload.setdefault("limit", None)
        payload.setdefault("sort_by", None)
        payload.setdefault("sort_direction", None)

        answer_type = payload["answer_type"]
        if not isinstance(answer_type, str):
            raise AnalysisPlanError(f"Unsupported answer type: {answer_type!r}")
        if answer_type != expected_type:
            raise AnalysisPlanError(
                f"Plan answer type {answer_type!r} does not match "
                f"expected type {expected_type!r}."
            )
        if answer_type not in _ANSWER_TYPES:
            raise AnalysisPlanError(f"Unsupported answer type: {answer_type!r}")

        table_name = payload["table"]
        if not isinstance(table_name, str) or table_name not in tables:
            raise AnalysisPlanError(f"Unknown source table: {table_name!r}")
        table = tables[table_name]

        aggregation = payload["aggregation"]
        if not isinstance(aggregation, str) or aggregation not in _AGGREGATIONS:
            raise AnalysisPlanError(f"Unsupported aggregation: {aggregation!r}")

        value_column = payload["value_column"]
        if value_column is not None and (
            not isinstance(value_column, str) or value_column not in table.columns
        ):
            raise AnalysisPlanError(f"Unknown value column: {value_column!r}")
        if aggregation in {"sum", "mean", "min", "max", "nunique"} and not value_column:
            raise AnalysisPlanError(
                f"Aggregation {aggregation!r} requires a value_column."
            )
        if aggregation in {"sum", "mean"} and not pd.api.types.is_numeric_dtype(
            table[value_column]
        ):
            raise AnalysisPlanError(
                f"Value column {value_column!r} is not numeric; use the code fallback."
            )

        date_column = payload["date_column"]
        start_date = payload["start_date"]
        end_date_exclusive = payload["end_date_exclusive"]
        date_values = (date_column, start_date, end_date_exclusive)
        if any(value is not None for value in date_values):
            if not all(value is not None for value in date_values):
                raise AnalysisPlanError(
                    "Date filtering requires date_column, start_date, and "
                    "end_date_exclusive."
                )
            if not isinstance(date_column, str) or date_column not in table.columns:
                raise AnalysisPlanError(f"Unknown date column: {date_column!r}")
            try:
                start = date.fromisoformat(start_date)
                end = date.fromisoformat(end_date_exclusive)
            except (TypeError, ValueError) as exc:
                raise AnalysisPlanError("Plan dates must use YYYY-MM-DD.") from exc
            if end <= start:
                raise AnalysisPlanError("end_date_exclusive must be after start_date.")
            parsed_dates = pd.to_datetime(table[date_column], errors="coerce")
            if not parsed_dates.notna().any():
                raise AnalysisPlanError(
                    f"Date column {date_column!r} contains no parseable dates."
                )

        filters = payload["filters"]
        if not isinstance(filters, list):
            raise AnalysisPlanError("filters must be a list.")
        validated_filters: List[Dict[str, Any]] = []
        for filter_spec in filters:
            if not isinstance(filter_spec, dict) or set(filter_spec) != {
                "column", "operator", "value"
            }:
                raise AnalysisPlanError(
                    "Each filter must contain exactly column, operator, and value."
                )
            column = filter_spec["column"]
            operator = filter_spec["operator"]
            value = filter_spec["value"]
            if not isinstance(column, str) or column not in table.columns:
                raise AnalysisPlanError(f"Unknown filter column: {column!r}")
            if not isinstance(operator, str) or operator not in _FILTER_OPERATORS:
                raise AnalysisPlanError(f"Unsupported filter operator: {operator!r}")
            if operator in {"eq", "ne"}:
                if not isinstance(value, (str, int, float, bool, type(None))):
                    raise AnalysisPlanError(f"Filter {operator!r} requires a scalar.")
                if not table[column].eq(value).any():
                    raise AnalysisPlanError(
                        f"Filter value {value!r} is absent from column {column!r}."
                    )
            elif operator in {"in", "not_in"}:
                if (
                    not isinstance(value, list)
                    or not value
                    or any(
                        not isinstance(item, (str, int, float, bool, type(None)))
                        for item in value
                    )
                ):
                    raise AnalysisPlanError(f"Filter {operator!r} requires a non-empty list.")
                if not all(table[column].eq(item).any() for item in value):
                    raise AnalysisPlanError(
                        f"One or more filter values are absent from column {column!r}."
                    )
            elif value is not None:
                raise AnalysisPlanError(
                    f"Filter {operator!r} requires value to be null."
                )
            validated_filters.append(dict(filter_spec))

        group_by = payload["group_by"]
        if (
            not isinstance(group_by, list)
            or len(group_by) > 2
            or any(not isinstance(column, str) for column in group_by)
        ):
            raise AnalysisPlanError("group_by must be a list of at most two columns.")
        if len(set(group_by)) != len(group_by):
            raise AnalysisPlanError("group_by columns must be unique.")
        if any(column not in table.columns for column in group_by):
            raise AnalysisPlanError("group_by references an unknown column.")

        output = payload["output"]
        if not isinstance(output, str) or output not in _OUTPUTS:
            raise AnalysisPlanError(f"Unsupported output mode: {output!r}")
        if output == "scalar" and group_by:
            raise AnalysisPlanError("Scalar output cannot include group_by columns.")
        if output != "scalar" and not group_by:
            raise AnalysisPlanError(f"Output mode {output!r} requires group_by columns.")
        if output == "group_keys" and answer_type not in {"string", "list"}:
            raise AnalysisPlanError("group_keys output requires string or list answer type.")
        if output == "group_keys" and len(group_by) != 1:
            raise AnalysisPlanError("group_keys output requires exactly one group_by column.")
        if output == "group_values" and answer_type != "list":
            raise AnalysisPlanError("group_values output requires list answer type.")
        if output == "table" and answer_type != "table":
            raise AnalysisPlanError("table output requires table answer type.")
        if output == "scalar" and answer_type != "number":
            raise AnalysisPlanError("Scalar aggregation currently supports number answers only.")

        limit = payload["limit"]
        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100
        ):
            raise AnalysisPlanError("limit must be an integer from 1 through 100 or null.")
        if output == "group_keys" and answer_type == "string" and limit != 1:
            raise AnalysisPlanError("String group_keys output requires limit=1.")

        sort_by = payload.get("sort_by")
        sort_direction = payload.get("sort_direction")

        if sort_by is None:
            sort_by = "value" if (group_by and aggregation in {"sum", "count", "nunique", "mean"}) else "key"
        elif not isinstance(sort_by, str) or sort_by not in {"key", "value"}:
            raise AnalysisPlanError("sort_by must be 'key' or 'value' or null.")

        if sort_direction is None:
            sort_direction = "desc" if sort_by == "value" else "asc"
        elif not isinstance(sort_direction, str) or sort_direction not in {"asc", "desc"}:
            raise AnalysisPlanError("sort_direction must be 'asc' or 'desc' or null.")

        if not group_by:
            sort_by = "key"
            sort_direction = "asc"
            limit = None

        return cls(
            answer_type=answer_type,
            table=table_name,
            aggregation=aggregation,
            value_column=value_column,
            date_column=date_column,
            start_date=start_date,
            end_date_exclusive=end_date_exclusive,
            filters=tuple(validated_filters),
            group_by=tuple(group_by),
            output=output,
            limit=limit,
            sort_by=sort_by,
            sort_direction=sort_direction,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_type": self.answer_type,
            "table": self.table,
            "aggregation": self.aggregation,
            "value_column": self.value_column,
            "date_column": self.date_column,
            "start_date": self.start_date,
            "end_date_exclusive": self.end_date_exclusive,
            "filters": list(self.filters),
            "group_by": list(self.group_by),
            "output": self.output,
            "limit": self.limit,
            "sort_by": self.sort_by,
            "sort_direction": self.sort_direction,
        }


def execute_analysis_plan(
    plan: AnalysisPlan,
    tables: Dict[str, pd.DataFrame],
) -> ExecutionResult:
    """Execute a validated plan using a restricted set of pandas operations."""
    result = ExecutionResult()
    result.code = json.dumps(plan.to_dict(), ensure_ascii=False)
    frame = tables[plan.table].copy()

    if plan.date_column is not None:
        dates = pd.to_datetime(frame[plan.date_column], errors="coerce")
        start = pd.Timestamp(plan.start_date)
        end = pd.Timestamp(plan.end_date_exclusive)
        frame = frame.loc[(dates >= start) & (dates < end)].copy()

    for filter_spec in plan.filters:
        series = frame[filter_spec["column"]]
        operator = filter_spec["operator"]
        value = filter_spec["value"]
        if operator == "eq":
            mask = series.eq(value)
        elif operator == "ne":
            mask = series.ne(value)
        elif operator == "in":
            mask = series.isin(value)
        elif operator == "not_in":
            mask = ~series.isin(value)
        elif operator == "is_null":
            mask = series.isna()
        else:
            mask = series.notna()
        frame = frame.loc[mask].copy()

    grouped = bool(plan.group_by)
    if not grouped and frame.empty and len(tables[plan.table]) > 0:
        if plan.filters or plan.date_column:
            raise AnalysisPlanError(
                "Plan filter conditions produced an empty DataFrame; use the code fallback."
            )

    if plan.aggregation == "count":
        if grouped:
            values = frame.groupby(list(plan.group_by), dropna=False).size()
        else:
            values = len(frame)
    elif plan.aggregation == "nunique":
        if grouped:
            values = frame.groupby(list(plan.group_by), dropna=False)[
                plan.value_column
            ].nunique()
        else:
            values = frame[plan.value_column].nunique()
    else:
        if grouped:
            values = frame.groupby(list(plan.group_by), dropna=False)[
                plan.value_column
            ].agg(plan.aggregation)
        else:
            values = getattr(frame[plan.value_column], plan.aggregation)()

    if grouped:
        if values.empty:
            raise AnalysisPlanError(
                "Grouped plan produced no rows; use the code fallback to re-evaluate."
            )
        if plan.sort_by == "value":
            values = values.sort_values(
                ascending=plan.sort_direction == "asc",
                kind="stable",
            )
        else:
            values = values.sort_index(
                ascending=plan.sort_direction == "asc",
                kind="stable",
            )
        if plan.limit is not None:
            values = values.head(plan.limit)

        if plan.output == "group_keys":
            keys = list(values.index)
            result.result = keys[0] if plan.answer_type == "string" and keys else keys
        elif plan.output == "group_values":
            result.result = values.tolist()
        else:
            rows = []
            index_values = values.index.tolist()
            if not isinstance(index_values, list):
                index_values = [index_values]
            for key, value in zip(index_values, values.tolist()):
                key_values = list(key) if isinstance(key, tuple) else [key]
                rows.append(key_values + [value])
            result.result = {
                "columns": list(plan.group_by) + [plan.value_column or "count"],
                "rows": rows,
            }
    else:
        result.result = values

    result.answer_type = plan.answer_type
    result.success = True
    return result
