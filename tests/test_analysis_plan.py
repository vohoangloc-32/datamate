import json

import pandas as pd
import pytest

from src.modules.analysis_plan import (
    AnalysisPlan,
    AnalysisPlanError,
    execute_analysis_plan,
)


@pytest.fixture
def tables():
    return {
        "sales": pd.DataFrame({
            "date": pd.to_datetime([
                "2025-01-01",
                "2025-01-02",
                "2025-01-03",
                "2025-01-03",
            ]),
            "status": ["done", "done", "cancelled", "done"],
            "order_id": [1, 2, 3, 2],
            "product": ["A", "B", "A", "A"],
            "amount": [10, 20, 30, 15],
        })
    }


def plan_payload(**overrides):
    payload = {
        "answer_type": "number",
        "table": "sales",
        "aggregation": "sum",
        "value_column": "amount",
        "date_column": "date",
        "start_date": "2025-01-02",
        "end_date_exclusive": "2025-01-04",
        "filters": [],
        "group_by": [],
        "output": "scalar",
        "limit": None,
        "sort_by": "key",
        "sort_direction": "asc",
    }
    payload.update(overrides)
    return payload


def parse_plan(tables, **overrides):
    return AnalysisPlan.from_response(
        json.dumps(plan_payload(**overrides)),
        tables,
        expected_type="number",
    )


def test_plan_executes_sum_with_start_inclusive_end_exclusive_dates(tables):
    plan = parse_plan(tables)

    result = execute_analysis_plan(plan, tables)

    assert result.success is True
    assert result.result == 65
    assert result.answer_type == "number"


def test_plan_executes_validated_category_filter(tables):
    plan = parse_plan(
        tables,
        filters=[{"column": "status", "operator": "eq", "value": "done"}],
    )

    result = execute_analysis_plan(plan, tables)

    assert result.result == 35


def test_plan_counts_distinct_order_identifiers(tables):
    plan = parse_plan(
        tables,
        aggregation="nunique",
        value_column="order_id",
        date_column=None,
        start_date=None,
        end_date_exclusive=None,
    )

    result = execute_analysis_plan(plan, tables)

    assert result.result == 3


def test_plan_rejects_filter_value_absent_from_column(tables):
    with pytest.raises(AnalysisPlanError, match="is absent"):
        parse_plan(
            tables,
            filters=[{"column": "status", "operator": "eq", "value": "completed"}],
        )


def test_plan_rejects_unknown_column_and_arbitrary_fields(tables):
    with pytest.raises(AnalysisPlanError, match="Unknown value column"):
        parse_plan(tables, value_column="revenue")

    payload = plan_payload(unrestricted_python="__import__('os')")
    with pytest.raises(AnalysisPlanError, match="Unknown plan fields"):
        AnalysisPlan.from_response(json.dumps(payload), tables, "number")


def test_plan_rejects_mismatched_answer_type(tables):
    with pytest.raises(AnalysisPlanError, match="does not match expected type"):
        AnalysisPlan.from_response(
            json.dumps(plan_payload(answer_type="percent")),
            tables,
            expected_type="number",
        )


def test_group_key_ranking_returns_flat_list(tables):
    payload = plan_payload(
        answer_type="list",
        aggregation="sum",
        group_by=["product"],
        output="group_keys",
        limit=2,
        sort_by="value",
        sort_direction="desc",
    )
    plan = AnalysisPlan.from_response(json.dumps(payload), tables, "list")

    result = execute_analysis_plan(plan, tables)

    assert result.result == ["A", "B"]
    assert all(not isinstance(value, (list, tuple, dict)) for value in result.result)


def test_grouped_values_are_flat_ordered_list(tables):
    payload = plan_payload(
        answer_type="list",
        aggregation="sum",
        group_by=["product"],
        output="group_values",
        sort_by="key",
        sort_direction="asc",
    )
    plan = AnalysisPlan.from_response(json.dumps(payload), tables, "list")

    result = execute_analysis_plan(plan, tables)

    assert result.result == [45, 20]


def test_grouped_table_output(tables):
    payload = plan_payload(
        answer_type="table",
        aggregation="count",
        value_column=None,
        group_by=["status"],
        output="table",
        sort_by="key",
    )
    plan = AnalysisPlan.from_response(json.dumps(payload), tables, "table")

    result = execute_analysis_plan(plan, tables)

    assert result.result == {
        "columns": ["status", "count"],
        "rows": [["cancelled", 1], ["done", 2]],
    }


def test_plan_handles_null_sort_and_reasoning_comments(tables):
    payload = plan_payload(
        sort_by=None,
        sort_direction=None,
        thought="I should sum the amounts for this date interval.",
        reason="Calculation based on sales records.",
    )
    plan = AnalysisPlan.from_response(json.dumps(payload), tables, "number")
    result = execute_analysis_plan(plan, tables)

    assert result.success is True
    assert result.result == 65


def test_plan_rejects_empty_dataframe_from_filters(tables):
    plan = parse_plan(
        tables,
        start_date="2025-05-01",
        end_date_exclusive="2025-05-30",
    )
    with pytest.raises(AnalysisPlanError, match="empty DataFrame"):
        execute_analysis_plan(plan, tables)

