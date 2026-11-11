import pandas as pd
import pytest

from src.modules.confidence_gate import ConfidenceGate
from src.modules.accounting_cleaner import patch_code_and_tables_for_rulebook
from src.modules.question_understanding import QuestionInfo
from src.modules.data_loader import load_business_data


def test_confidence_gate_mild_penalty_for_valid_percent_retry():
    """
    Phase 12: Percent answers with 1 execution retry in valid range (0 < val < 100)
    receive mild penalty (-0.15) instead of severe penalty (-0.35), keeping confidence >= 0.45.
    """
    gate = ConfidenceGate(min_confidence=0.45)
    q_info = QuestionInfo(expected_type="percent", confidence=0.9)

    # 1 retry with plausible percent (e.g. 14.52%) -> should PASS (confidence 1.0 - 0.25 - 0.15 = 0.60 >= 0.45)
    ans_valid = {"type": "percent", "answer": 14.52}
    res_valid = gate.evaluate(ans_valid, question_info=q_info, execution_retries=1)
    assert res_valid["type"] == "percent"
    assert res_valid["answer"] == 14.52

    # 2 retries with plausible percent -> should ABSTAIN (flaky execution: 1.0 - 0.50 - 0.35 = 0.15 < 0.45)
    res_2retries = gate.evaluate(ans_valid, question_info=q_info, execution_retries=2)
    assert res_2retries["type"] == "abstain"

    # 1 retry with zero percent (degenerate: 1.0 - 0.25 - 0.35 = 0.40 < 0.45) -> should ABSTAIN
    ans_zero = {"type": "percent", "answer": 0.0}
    res_zero = gate.evaluate(ans_zero, question_info=q_info, execution_retries=1)
    assert res_zero["type"] == "abstain"


def test_accounting_cleaner_neutralizes_manual_ratio_multiplication():
    """
    Phase 12: Safe conversion neutralization.
    When precomputed 'Số lượng quy đổi' exists, neutralize LLM's buggy manual
    multiplication that blindly scales base-unit rows by conversion ratios.
    """
    tables = {
        "don_hang": pd.DataFrame({
            "Mã đơn": [1, 2],
            "Tên SP": ["Bánh croissant", "Cà phê"],
            "Số lượng": [14, 5],
            "Số lượng quy đổi": [14, 5],
        }),
        "danh_sach_quy_doi": pd.DataFrame({
            "Tên SP": ["Bánh croissant"],
            "Tỉ lệ quy đổi": [6],
        }),
    }
    code = """
merged = don_hang.merge(danh_sach_quy_doi, on='Tên SP', how='left')
merged['Số lượng quy đổi'] = merged['Số lượng'] * merged['Tỉ lệ quy đổi'].fillna(1)
result = merged.groupby('Tên SP')['Số lượng quy đổi'].sum()
"""
    patched, _ = patch_code_and_tables_for_rulebook(
        code=code,
        tables=tables,
        question="thang 6/2025 top 5 hang ban chay nhat la j z",
    )
    # The manual multiplication line must be neutralized into a comment
    assert "neutralized in favor of pre-computed Số lượng quy đổi" in patched


def test_accounting_cleaner_ranking_redirects_to_quy_doi_even_with_ratio_in_code():
    """
    Phase 12: Product ranking queries redirect raw quantity to 'Số lượng quy đổi'
    even if 'Tỉ lệ quy đổi' or 'Hệ số' is mentioned in the code.
    """
    tables = {
        "don_hang": pd.DataFrame({
            "Mã đơn": [1, 2],
            "Tên SP": ["A", "B"],
            "Số lượng": [10, 20],
            "Số lượng quy đổi": [100, 20],
        }),
    }
    code = """
# Reference to Tỉ lệ quy đổi mentioned here
grouped = don_hang.groupby('Tên SP')['Số lượng'].sum().reset_index()
result = grouped.sort_values(by='Số lượng', ascending=False)
"""
    patched, _ = patch_code_and_tables_for_rulebook(
        code=code,
        tables=tables,
        question="check gium cho toi 5 mon ban chay nhat trong thang 4/2025",
    )
    assert "['Số lượng quy đổi']" in patched
    assert "['Số lượng']" not in patched


def test_accounting_cleaner_pharmacy_active_ingredient_exact_match():
    """
    Phase 12: Rulebook §6 lines 89-91 active ingredient exact match.
    Convert .str.contains() to exact equality on active ingredient columns.
    """
    tables = {
        "san_pham": pd.DataFrame({
            "MaSP": [1, 2, 3],
            "ThanhPhanChinh": ["Vitamin C", "Sắt + Vitamin C", "Vitamin C 500mg"],
        }),
    }
    code = """
filtered = san_pham[san_pham['ThanhPhanChinh'].str.contains('Vitamin C', na=False)]
result = len(filtered)
"""
    patched, _ = patch_code_and_tables_for_rulebook(
        code=code,
        tables=tables,
        question="Doanh thu thuốc hoạt chất Vitamin C tháng này là bao nhiêu?",
    )
    assert ".astype(str).str.strip().str.lower() == 'vitamin c'" in patched


def test_accounting_cleaner_delivery_trips_dedup():
    """
    Phase 12: Rulebook §6 lines 95-97 delivery trips deduplication.
    Convert nunique/len on trips to drop_duplicates on [Mã phiếu, Lượt giao].
    """
    tables = {
        "so_giao_hang": pd.DataFrame({
            "Mã phiếu": ["P1", "P1", "P2"],
            "Lượt giao": [1, 1, 1],
        }),
    }
    code = """
result = so_giao_hang['Lượt giao'].nunique()
"""
    patched, _ = patch_code_and_tables_for_rulebook(
        code=code,
        tables=tables,
        question="tháng 3/2026, xe chở được bao nhiêu đợt hàng",
    )
    assert "drop_duplicates(subset=['Mã phiếu', 'Lượt giao']).shape[0]" in patched


def test_data_loader_removes_monthly_sheet_components_after_combining(tmp_path):
    """
    Phase 12: Monthly sheet components are popped from tables once combined into
    __monthly_combined, preventing the LLM from concatenating sub-sheets + combined table.
    """
    biz_dir = tmp_path / "biz_split_sheets"
    biz_dir.mkdir()
    xlsx_file = biz_dir / "sales.xlsx"
    with pd.ExcelWriter(xlsx_file) as writer:
        pd.DataFrame({"order_id": [1], "amount": [100]}).to_excel(writer, sheet_name="T01_2025", index=False)
        pd.DataFrame({"order_id": [2], "amount": [200]}).to_excel(writer, sheet_name="T02_2025", index=False)

    tables = load_business_data(str(biz_dir), use_cache=False)
    assert "sales__monthly_combined" in tables
    assert len(tables["sales__monthly_combined"]) == 2
    # Individual sheets must NOT be present in tables to avoid 2x double counting
    assert "sales__T01_2025" not in tables
    assert "sales__T02_2025" not in tables
