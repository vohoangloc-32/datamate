"""
Tests for Phase 13 Architecture Improvements:
- Safe conversion merge patching (no repeated how= keyword even with parentheses in column names)
- 2D list entity vs metric unpacking in CodeExecutor
- Standardized percentage rounding with Decimal ROUND_HALF_UP in format_percent
- Rulebook §1, §2, §4 guidance in format_accounting_context
- Semantic feedback in RetryEngine for degenerate percentage and debt repayments
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock

from src.modules.accounting_cleaner import (
    format_accounting_context,
    patch_code_and_tables_for_rulebook,
)
from src.modules.answer_formatter import format_percent
from src.modules.code_executor import CodeExecutor
from src.modules.retry_engine import RetryEngine


def test_merge_conversion_no_repeated_how():
    """Verify that conversion table merge patch never generates repeated how= keyword."""
    code_with_left = """
merged_data = filtered_data.merge(
    DanhSachQuyDoi,
    left_on=['Nội dung', 'Đơn vị (tính)'],
    right_on=['Tên hàng', 'Đơn vị quy đổi'],
    how='left'
)
total_kg = (merged_data['so_luong'] * merged_data['Tỉ lệ quy đổi']).sum()
result = total_kg
answer_type = "number"
"""
    patched, _ = patch_code_and_tables_for_rulebook(code_with_left, "tính tổng kg", {})
    # Compile must succeed without SyntaxError
    compiled = compile(patched, "<string>", "exec")
    assert compiled is not None
    # how='left' should appear exactly once in the merge
    assert patched.count("how='left'") + patched.count('how="left"') == 1

    code_with_inner = """
merged_data = filtered_data.merge(
    DanhSachQuyDoi,
    left_on=['Nội dung', 'Đơn vị (tính)'],
    right_on=['Tên hàng', 'Đơn vị quy đổi'],
    how='inner'
)
"""
    patched2, _ = patch_code_and_tables_for_rulebook(code_with_inner, "tính tổng kg", {})
    assert "how='left'" in patched2
    assert "how='inner'" not in patched2
    compile(patched2, "<string>", "exec")

    code_without_how = """
merged_data = filtered_data.merge(
    DanhSachQuyDoi,
    left_on=['Nội dung', 'Đơn vị (tính)'],
    right_on=['Tên hàng', 'Đơn vị quy đổi']
)
"""
    patched3, _ = patch_code_and_tables_for_rulebook(code_without_how, "tính tổng kg", {})
    assert "how='left'" in patched3
    compile(patched3, "<string>", "exec")


def test_entity_list_unpacking():
    """Verify CodeExecutor unpacks entity names for entity questions and metrics for metric questions."""
    executor = CodeExecutor()

    # Case 1: Question asks for top customers, result is list of [name, amount]
    code_customers = """
result = [['La Thảo Ngân', 13084500], ['Ngô Minh Dũng', 8011500], ['Vũ Hồng Anh', 7692000]]
answer_type = 'list'
"""
    res1 = executor.execute(
        code_customers,
        {},
        as_of_date="2026-07-28",
        question="Cho tôi 3 khách mua nhiều tiền nhất tháng 4/2026.",
    )
    assert res1.result == ["La Thảo Ngân", "Ngô Minh Dũng", "Vũ Hồng Anh"]

    # Case 2: Question asks for revenue per month, result is list of [month, amount]
    code_monthly = """
result = [['2025-04', 287435200], ['2025-05', 346554700], ['2025-06', 309693800]]
answer_type = 'list'
"""
    res2 = executor.execute(
        code_monthly,
        {},
        as_of_date="2026-07-28",
        question="dthu 3 tháng liền nhau kết thúc ở tháng 6/2025 là bao nhiêu mỗi tháng?",
    )
    assert res2.result == [287435200, 346554700, 309693800]


def test_format_percent_round_half_up():
    """Verify that format_percent uses arithmetic rounding (ROUND_HALF_UP)."""
    assert format_percent(1.205) == 1.21
    assert format_percent(1.955) == 1.96
    assert format_percent(3.454) == 3.45
    assert format_percent(104.95) == 104.95
    assert format_percent(-50.0) == -50.0
    assert format_percent(None) is None


def test_accounting_context_rulebook_guidance():
    """Verify Rulebook §1, §2, §4 notes in format_accounting_context."""
    # Work shift
    ctx_shift = format_accounting_context({}, "Tuần trước riêng ca sáng bán được bao nhiêu?")
    assert ctx_shift is not None
    assert "Ca sáng, ca chiều (Rulebook §2 line 33)" in ctx_shift

    # Lunar Tet
    ctx_tet = format_accounting_context({}, "Dịp Tết âm lịch 2026 có doanh thu hơn trước Tết?")
    assert ctx_tet is not None
    assert "Tết, dịp Tết (Rulebook §2 line 32)" in ctx_tet

    # Category proportion
    ctx_pct = format_accounting_context({}, "tỉ trọng của nhóm sữa trong dthu tháng 1/2026 là bnhiu %")
    assert ctx_pct is not None
    assert "Tỉ trọng danh mục/nhóm hàng (Rulebook §1)" in ctx_pct

    # Debt collection
    ctx_debt = format_accounting_context({}, "tháng 8/2025 cửa hàng thu được bao nhiêu tiền khách trả nợ?")
    assert ctx_debt is not None
    assert "Công nợ phát sinh và trả nợ trong kỳ (Rulebook §4 line 58)" in ctx_debt


def test_retry_engine_semantic_validation():
    """Verify RetryEngine intercepts degenerate 100% on proportion questions."""
    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build_retry_prompt.return_value = "new prompt"

    engine = RetryEngine(
        code_executor=MagicMock(),
        prompt_builder=mock_prompt_builder,
        llm_generate_fn=MagicMock(return_value="result = 6.04\nanswer_type = 'percent'"),
        max_retries=1,
    )

    # First attempt returns 100.0%, second attempt returns 6.04%
    call_count = 0
    def mock_generate(prompt):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return "result = 100.0\nanswer_type = 'percent'"
        return "result = 6.04\nanswer_type = 'percent'"

    engine.llm_generate = mock_generate
    engine.executor = CodeExecutor()

    res, retries = engine.run(
        prompt="Initial prompt",
        tables={},
        as_of_date="2026-08-01",
        question="tỉ trọng của nhóm sữa trong dthu tháng 1/2026 là bnhiu %",
    )

    assert retries == 1
    assert res.result == 6.04
    assert mock_prompt_builder.build_retry_prompt.called
    error_arg = mock_prompt_builder.build_retry_prompt.call_args[0][2]
    assert "SemanticWarning: Result produced 100.0% for a category proportion question" in error_arg
