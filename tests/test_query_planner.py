import json

import pandas as pd

from src.modules.query_planner import build_query_plan


def test_query_plan_validates_real_tables_and_columns():
    plan = build_query_plan(
        lambda _: json.dumps({
            "metric": "distinct orders",
            "aggregation": "count_distinct",
            "entity_column": "order_id",
            "date_column": "order_date",
            "start_date": "2025-08-23",
            "end_date": "2025-08-24",
            "filters": [],
            "group_by": [],
            "source_tables": ["orders"],
            "answer_type": "number",
            "confidence": 0.9,
            "assumptions": [],
            "reason": "count order IDs",
        }),
        question="Hôm qua có bao nhiêu đơn hàng?",
        as_of_date="2025-08-24",
        schema="orders(order_id, order_date)",
        rulebook="",
        tables={"orders": pd.DataFrame({"order_id": [1], "order_date": ["2025-08-23"]})},
        table_var_names={"orders": "orders"},
        expected_type="number",
    )

    assert plan["status"] == "validated"
    assert plan["aggregation"] == "count_distinct"


def test_invalid_query_plan_falls_back_instead_of_inventing_data():
    plan = build_query_plan(
        lambda _: '{"source_tables":["missing"],"entity_column":"missing","confidence":0.9}',
        question="Tổng tiền?",
        as_of_date="2025-08-24",
        schema="orders(amount)",
        rulebook="",
        tables={"orders": pd.DataFrame({"amount": [1]})},
        table_var_names={"orders": "orders"},
        expected_type="number",
    )

    assert plan["status"] == "fallback"
    assert "invalid" in plan["reason"]


def test_non_unanswerable_plan_without_source_table_is_rejected():
    plan = build_query_plan(
        lambda _: json.dumps({
            "aggregation": "sum",
            "answer_type": "number",
            "confidence": 0.9,
        }),
        question="Tổng tiền?",
        as_of_date="2025-08-24",
        schema="orders(amount)",
        rulebook="",
        tables={"orders": pd.DataFrame({"amount": [1]})},
        table_var_names={"orders": "orders"},
        expected_type="number",
    )

    assert plan["status"] == "fallback"


def test_order_count_requires_distinct_order_ids():
    plan = build_query_plan(
        lambda _: json.dumps({
            "aggregation": "count",
            "entity_column": "order_id",
            "source_tables": ["orders"],
            "answer_type": "number",
            "confidence": 0.9,
        }),
        question="Có bao nhiêu đơn hàng?",
        as_of_date="2025-08-24",
        schema="orders(order_id)",
        rulebook="",
        tables={"orders": pd.DataFrame({"order_id": [1, 1]})},
        table_var_names={"orders": "orders"},
        expected_type="number",
    )

    assert plan["status"] == "fallback"
    assert "distinct" in plan["reason"]
