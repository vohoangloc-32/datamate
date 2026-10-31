"""
Tests for code_executor module.
"""

import pytest
import pandas as pd
from datetime import date
from src.modules.code_executor import CodeExecutor, _extract_code_from_response


@pytest.fixture
def sample_tables():
    orders = pd.DataFrame({
        "order_id": [1, 2, 3],
        "customer": ["An", "Binh", "Chi"],
        "amount": [100000, 250000, 150000],
        "status": ["Completed", "Completed", "Cancelled"],
    })
    return {"orders": orders}


def test_extract_code_from_response():
    resp_with_markdown = """Here is the Python code to compute the revenue:
```python
result = orders['amount'].sum()
answer_type = "number"
```
Explanation: We sum the amount column.
"""
    extracted = _extract_code_from_response(resp_with_markdown)
    assert extracted == "result = orders['amount'].sum()\nanswer_type = \"number\""

    # Raw code without markdown block
    raw_code = "result = 42\nanswer_type = 'number'"
    assert _extract_code_from_response(raw_code) == raw_code


def test_execute_success(sample_tables):
    executor = CodeExecutor(timeout_seconds=5)
    code = """
valid_orders = orders[orders['status'] != 'Cancelled']
result = int(valid_orders['amount'].sum())
answer_type = "number"
"""
    res = executor.execute(code, sample_tables, "2024-05-18", extract_code=False)
    assert res.success is True
    assert res.result == 350000
    assert res.answer_type == "number"
    assert res.error is None


def test_execute_with_markdown(sample_tables):
    executor = CodeExecutor(timeout_seconds=5)
    resp = """```python
top_cust = orders.sort_values(by='amount', ascending=False).iloc[0]['customer']
result = str(top_cust)
answer_type = "string"
```"""
    res = executor.execute(resp, sample_tables, "2024-05-18", extract_code=True)
    assert res.success is True
    assert res.result == "Binh"
    assert res.answer_type == "string"


def test_execute_handles_error(sample_tables):
    executor = CodeExecutor(timeout_seconds=5)
    # KeyError on non-existent column
    code = """
result = orders['non_existent_column'].sum()
answer_type = "number"
"""
    res = executor.execute(code, sample_tables, "2024-05-18", extract_code=False)
    assert res.success is False
    assert res.error is not None
    assert "KeyError" in res.error


def test_execute_captures_stdout(sample_tables):
    executor = CodeExecutor(timeout_seconds=5)
    code = """
print("Debug log message")
result = 123
answer_type = "number"
"""
    res = executor.execute(code, sample_tables, "2024-05-18", extract_code=False)
    assert res.success is True
    assert "Debug log message" in res.stdout


def test_execute_single_table_df_alias(sample_tables):
    executor = CodeExecutor(timeout_seconds=5)
    # When single table is loaded, it should be available as `df` too
    code = """
result = len(df)
answer_type = "number"
"""
    res = executor.execute(code, sample_tables, "2024-05-18", extract_code=False)
    assert res.success is True
    assert res.result == 3


def test_execute_allows_whitelisted_import(sample_tables):
    executor = CodeExecutor(timeout_seconds=5)
    code = """
import pandas as pd
import numpy as np
from datetime import date
import re
result = int(np.sum(orders['amount']))
answer_type = "number"
"""
    res = executor.execute(code, sample_tables, "2024-05-18", extract_code=False)
    assert res.success is True
    assert res.result == 500000


def test_execute_blocks_dangerous_import(sample_tables):
    executor = CodeExecutor(timeout_seconds=5)
    code = """
import os
result = os.getcwd()
answer_type = "string"
"""
    res = executor.execute(code, sample_tables, "2024-05-18", extract_code=False)
    assert res.success is False
    assert "ImportError" in res.error

