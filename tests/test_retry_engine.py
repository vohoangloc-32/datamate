"""Tests for schema-aware retries and generation failures."""

import pandas as pd

from src.modules.code_executor import CodeExecutor
from src.modules.prompt_builder import PromptBuilder
from src.modules.retry_engine import RetryEngine


def test_name_error_retry_lists_the_real_dataframe_variables():
    prompts = []
    responses = iter([
        "orders_df['amount'] = orders_df['amount']\nresult = 1\nanswer_type = 'number'",
        "result = float(orders['amount'].sum())\nanswer_type = 'number'",
    ])

    def generate(prompt):
        prompts.append(prompt)
        return next(responses)

    engine = RetryEngine(
        code_executor=CodeExecutor(),
        prompt_builder=PromptBuilder(few_shot_examples=[]),
        llm_generate_fn=generate,
        max_retries=2,
    )
    tables = {"orders": pd.DataFrame({"amount": [10, 20]})}

    result, retries = engine.run(
        prompt="Use the provided tables. Question: total amount?",
        tables=tables,
        as_of_date="2025-01-01",
        expected_type="number",
    )

    assert result.success is True
    assert result.result == 30.0
    assert retries == 1
    assert len(prompts) == 2
    assert "AVAILABLE VARIABLES AND COLUMNS" in prompts[1]
    assert "`df`" in prompts[1]
    assert "`orders`" in prompts[1]


def test_context_overflow_does_not_repeat_the_same_generation_request():
    calls = []

    def generate(prompt):
        calls.append(prompt)
        raise RuntimeError(
            "VLLMValidationError: maximum context length exceeded; input_tokens=16385"
        )

    engine = RetryEngine(
        code_executor=CodeExecutor(),
        prompt_builder=PromptBuilder(few_shot_examples=[]),
        llm_generate_fn=generate,
        max_retries=3,
    )

    result, retries = engine.run(
        prompt="A prompt which the generator rejects.",
        tables={},
        as_of_date="2025-01-01",
    )

    assert len(calls) == 1
    assert retries == 0
    assert result.answer_type == "abstain"
    assert result.error is not None
    assert "maximum context length" in result.error


def test_file_not_found_retry_context_requires_using_preloaded_tables():
    engine = RetryEngine(
        code_executor=CodeExecutor(),
        prompt_builder=PromptBuilder(few_shot_examples=[]),
        llm_generate_fn=lambda prompt: "",
    )
    tables = {"sales": pd.DataFrame({"Ngày bán": ["2025-01-01"], "Doanh thu": [10]})}

    context = engine._available_context(
        tables,
        "FileNotFoundError: 'sales.xlsx' does not exist",
    )

    assert "Do not read or reload files" in context
    assert "`df` (table `sales`)" in context
    assert "Ngày bán" in context
    assert "Doanh thu" in context


def test_datetime_type_error_retry_context_lists_datetime_dtypes():
    engine = RetryEngine(
        code_executor=CodeExecutor(),
        prompt_builder=PromptBuilder(few_shot_examples=[]),
        llm_generate_fn=lambda prompt: "",
    )
    tables = {
        "sales": pd.DataFrame({
            "order_date": pd.to_datetime(["2025-01-01"]),
            "Doanh thu": [10],
        }),
    }

    context = engine._available_context(
        tables,
        "TypeError: Cannot compare Timestamp with datetime.date",
    )

    assert "pd.to_datetime(series, errors='coerce')" in context
    assert "pd.Timestamp(...)" in context
    assert ".dt.date" in context
    assert "datetime64[" in context


def test_dt_date_usage_retries_with_timestamp_guidance():
    prompts = []
    responses = iter([
        "orders['order_date'] = pd.to_datetime(['2025-01-01'] * len(orders))\n"
        "orders = orders[orders['order_date'].dt.date == date(2025, 1, 1)]\n"
        "result = len(orders)\nanswer_type = 'number'",
        "orders['order_date'] = pd.to_datetime(['2025-01-01'] * len(orders))\n"
        "orders = orders[orders['order_date'] == pd.Timestamp('2025-01-01')]\n"
        "result = len(orders)\nanswer_type = 'number'",
    ])

    def generate(prompt):
        prompts.append(prompt)
        return next(responses)

    engine = RetryEngine(
        code_executor=CodeExecutor(),
        prompt_builder=PromptBuilder(few_shot_examples=[]),
        llm_generate_fn=generate,
        max_retries=1,
    )
    result, retries = engine.run(
        prompt="Use the provided tables. Question: number of orders today?",
        tables={"orders": pd.DataFrame({"order_date": ["2025-01-01"]})},
        as_of_date="2025-01-01",
        expected_type="number",
    )

    assert result.success is True
    assert result.result == 1
    assert retries == 1
    assert len(prompts) == 2
    assert "keep values as pandas timestamps" in prompts[1]
    assert "pd.Timestamp(...)" in prompts[1]


def test_repeated_code_and_error_stops_retry_and_includes_file_feedback():
    prompts = []
    code = "result = pd.read_excel('sales.xlsx').shape[0]\nanswer_type = 'number'"

    def generate(prompt):
        prompts.append(prompt)
        return code

    engine = RetryEngine(
        code_executor=CodeExecutor(),
        prompt_builder=PromptBuilder(few_shot_examples=[]),
        llm_generate_fn=generate,
        max_retries=3,
    )
    tables = {"sales": pd.DataFrame({"amount": [10]})}

    result, retries = engine.run(
        prompt="Use the provided tables. Question: row count?",
        tables=tables,
        as_of_date="2025-01-01",
        expected_type="number",
    )

    assert result.success is False
    assert result.answer_type == "abstain"
    assert retries == 1
    assert len(prompts) == 2
    assert "FileAccessError" in prompts[1]
    assert "Do not read or reload files" in prompts[1]
    assert "amount" in prompts[1]


def test_syntax_error_retry_includes_valid_variables_and_syntax_guidance():
    prompts = []
    responses = iter([
        "sales.2026 = 1\nresult = 1\nanswer_type = 'number'",
        "result = sales['amount'].sum()\nanswer_type = 'number'",
    ])

    def generate(prompt):
        prompts.append(prompt)
        return next(responses)

    engine = RetryEngine(
        code_executor=CodeExecutor(),
        prompt_builder=PromptBuilder(few_shot_examples=[]),
        llm_generate_fn=generate,
        max_retries=1,
    )
    result, retries = engine.run(
        prompt="Use the provided tables. Question: total amount?",
        tables={"sales": pd.DataFrame({"amount": [10, 20]})},
        as_of_date="2025-01-01",
        expected_type="number",
    )

    assert result.success is True
    assert result.result == 30
    assert retries == 1
    assert "syntax error" in prompts[1]
    assert "Available DataFrame variables and actual columns" in prompts[1]


def test_distinct_retries_continue_after_first_failure():
    prompts = []
    responses = iter([
        "result = missing_table['amount'].sum()\nanswer_type = 'number'",
        "result = sales['missing_column'].sum()\nanswer_type = 'number'",
        "result = sales['amount'].sum()\nanswer_type = 'number'",
    ])

    def generate(prompt):
        prompts.append(prompt)
        return next(responses)

    engine = RetryEngine(
        code_executor=CodeExecutor(),
        prompt_builder=PromptBuilder(few_shot_examples=[]),
        llm_generate_fn=generate,
        max_retries=2,
    )
    tables = {"sales": pd.DataFrame({"amount": [10, 20]})}

    result, retries = engine.run(
        prompt="Use the provided tables. Question: total amount?",
        tables=tables,
        as_of_date="2025-01-01",
        expected_type="number",
    )

    assert result.success is True
    assert result.result == 30
    assert retries == 2
    assert len(prompts) == 3
    assert "AVAILABLE VARIABLES AND COLUMNS" in prompts[1]
    assert "amount" in prompts[2]
