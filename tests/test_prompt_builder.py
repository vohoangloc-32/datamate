"""Tests for prompt construction and tokenizer-aware prompt budgets."""

import pandas as pd
import pytest

from src.modules.prompt_builder import PromptBuilder, PromptTokenLimitError
from src.modules.question_understanding import QuestionInfo
from src.modules.schema_registry import SchemaRegistry


def _prompt_inputs():
    schema = SchemaRegistry().build_from_dict({
        "sales": pd.DataFrame({"order_date": ["2025-01-01"], "amount": [100]}),
    })
    return {
        "question": "Tổng doanh thu tháng 1 là bao nhiêu?",
        "as_of_date": "2025-02-01",
        "schema": schema,
        "table_var_names": {"sales": "sales"},
        "head_rows": {"sales": "sample row " * 2000},
    }


def test_build_compacts_optional_sections_to_fit_token_budget():
    examples = [{
        "expected_type": "number",
        "question": "example " * 1000,
        "code": "result = 1\nanswer_type = 'number'",
        "answer": "1",
    }]
    builder = PromptBuilder(few_shot_examples=examples)
    prompt = builder.build(
        **_prompt_inputs(),
        token_counter=len,
        max_tokens=5000,
    )

    assert len(prompt) <= 5000
    assert "Question: Tổng doanh thu tháng 1 là bao nhiêu?" in prompt
    assert "order_date" in prompt
    assert "SAMPLE ROWS" not in prompt
    assert "## EXAMPLES" not in prompt


def test_build_raises_when_required_prompt_cannot_fit():
    builder = PromptBuilder(few_shot_examples=[])
    with pytest.raises(PromptTokenLimitError, match="Core prompt exceeds"):
        builder.build(
            **_prompt_inputs(),
            token_counter=len,
            max_tokens=100,
        )


def test_prompt_marks_available_schema_authoritative_and_forbids_file_reads():
    builder = PromptBuilder(few_shot_examples=[{
        "expected_type": "number",
        "question": "Example question",
        "code": "result = 1\nanswer_type = 'number'",
        "answer": "1",
    }])

    prompt = builder.build(
        **_prompt_inputs(),
        question_info=QuestionInfo(expected_type="number"),
    )

    assert "Use only the exact variable names and column names listed in AVAILABLE DATA" in prompt
    assert "Do not read or reload data from files" in prompt
    assert "Preserve every explicit question entity/filter" in prompt
    assert "combine all matching tables" in prompt
    assert "Order counts use distinct order IDs" in prompt
    assert "Use actual status values in the data" in prompt
    assert "For a status rate" in prompt
    assert "half-open date ranges" in prompt
    assert "structured plan and rulebook" in prompt
    assert "empty result from unavailable data" in prompt
    assert "system failures from unavailable data" in prompt
    assert "nunique()" in prompt
    assert "Their table variables and columns are not available unless listed in AVAILABLE DATA" in prompt


def test_prompt_includes_structured_query_plan():
    prompt = PromptBuilder(few_shot_examples=[]).build(
        **_prompt_inputs(),
        query_plan={
            "status": "validated",
            "aggregation": "count_distinct",
            "entity_column": "order_id",
            "source_tables": ["orders"],
        },
    )

    assert "STRUCTURED QUERY PLAN" in prompt
    assert "count_distinct" in prompt


def test_default_examples_never_convert_datetime_columns_to_python_dates():
    builder = PromptBuilder()

    assert builder.few_shot_examples
    assert all(".dt.date" not in example["code"] for example in builder.few_shot_examples)
    assert all("date(" not in example["code"] for example in builder.few_shot_examples)


def test_examples_with_unavailable_variables_or_columns_are_omitted():
    builder = PromptBuilder()

    prompt = builder.build(**_prompt_inputs())

    assert "## EXAMPLES" not in prompt


def test_retry_prompt_compacts_context_and_preserves_error_and_question():
    builder = PromptBuilder(few_shot_examples=[])
    original = builder.build(
        **_prompt_inputs(),
    )
    retry = builder.build_retry_prompt(
        original,
        generated_code="orders_df['order_date'] = 1\n" * 1000,
        error_message="KeyError: 'order_date'\n" + ("traceback " * 1000),
        available_context="Available columns: order_date, amount\n" + ("column " * 1000),
        token_counter=len,
        max_tokens=5000,
    )

    assert len(retry) <= 5000
    assert "Question: Tổng doanh thu tháng 1 là bao nhiêu?" in retry
    assert "KeyError: 'order_date'" in retry
    assert "## AVAILABLE VARIABLES AND COLUMNS" in retry
