"""
test_phase6_architecture.py — Verification tests for Phase 6 Architectural Improvements:
1. Data cleaning: internal whitespace normalization & entity casing canonicalization
2. Excel multi-tier subheaders & missing year resolution from sheet period
3. General relational bridging: FK aliasing, temporal & dimension propagation
4. Sandbox merge suffix collision handling
5. Non-leap year Feb 29 date clamping
6. List tuple unpacking & snapshot table confidence penalty exemption
"""

import pandas as pd
import pytest

from src.modules.data_loader import (
    _coerce_types,
    _coerce_date_column,
    _bridge_relational_tables,
    _compact_col_name,
)
from src.modules.code_executor import CodeExecutor
from src.modules.accounting_cleaner import patch_code_and_tables_for_rulebook
from src.modules.confidence_gate import ConfidenceGate
from src.modules.question_understanding import QuestionInfo


def test_internal_whitespace_and_case_canonicalization():
    df = pd.DataFrame({
        "Tên sản phẩm": [
            "Gaviscon  hộp  24 gói",
            "Gaviscon hộp 24 gói",
            "Gaviscon hộp 24 gói",
            "GAVISCON HỘP 24 GÓI",
            "Panadol  cảm cúm",
        ]
    })
    df_clean = _coerce_types(df)
    # Whitespace collapsed
    assert df_clean["Tên sản phẩm"].iloc[0] == "Gaviscon hộp 24 gói"
    # Case harmonized to modal casing
    assert df_clean["Tên sản phẩm"].iloc[3] == "Gaviscon hộp 24 gói"
    assert df_clean["Tên sản phẩm"].iloc[4] == "Panadol cảm cúm"


def test_missing_year_date_coercion():
    s = pd.Series(["1/1", "15/01", "01/01/2025"])
    parsed = _coerce_date_column(s, default_year=2025)
    assert parsed is not None
    assert str(parsed.iloc[0].date()) == "2025-01-01"
    assert str(parsed.iloc[1].date()) == "2025-01-15"
    assert str(parsed.iloc[2].date()) == "2025-01-01"


def test_compact_col_name_recognizes_vietnamese_subheaders():
    assert _compact_col_name("Đơn") in {"hang", "don", "tien"}
    assert _compact_col_name("Hàng") in {"hang", "don", "tien"}
    assert _compact_col_name("Tiền") in {"hang", "don", "tien"}


def test_relational_bridging_fk_aliasing_and_propagation():
    df_header = pd.DataFrame({
        "Mã chứng từ": ["HD01", "HD02", "HD03"],
        "Thời gian": pd.to_datetime(["2026-03-01", "2026-03-02", "2026-03-03"]),
        "Tổng tiền hàng": [100000, 200000, 300000],
    })
    df_detail = pd.DataFrame({
        "Ma don": ["HD01", "HD01", "HD02", "HD03"],
        "Mã SKU": ["SKU1", "SKU2", "SKU1", "SKU3"],
        "Thành tiền": [40000, 60000, 200000, 300000],
    })
    df_catalog = pd.DataFrame({
        "Mã SKU": ["SKU1", "SKU2", "SKU3"],
        "Tên sản phẩm": ["Thuốc A", "Thuốc B", "Thuốc C"],
    })

    tables = {
        "don_hang": df_header,
        "chi_tiet": df_detail,
        "san_pham": df_catalog,
    }

    bridged = _bridge_relational_tables(tables)

    # 1. FK aliasing: both tables have both key names
    assert "Ma don" in bridged["don_hang"].columns
    assert "Mã chứng từ" in bridged["chi_tiet"].columns

    # 2. Temporal propagation: detail table gets transaction date
    assert "Thời gian" in bridged["chi_tiet"].columns
    assert str(bridged["chi_tiet"]["Thời gian"].iloc[0].date()) == "2026-03-01"

    # 3. Dimension propagation: detail table gets product name
    assert "Tên sản phẩm" in bridged["chi_tiet"].columns
    assert bridged["chi_tiet"]["Tên sản phẩm"].iloc[0] == "Thuốc A"


def test_merge_suffix_collision_resolution():
    df1 = pd.DataFrame({
        "id": [1, 2],
        "Trạng thái": ["Hoàn thành", "Hoàn thành"],
        "Doanh số": [100, 200],
    })
    df2 = pd.DataFrame({
        "id": [1, 2],
        "Trạng thái": ["Active", "Active"],
        "Tên khách": ["A", "B"],
    })
    tables = {"df1": df1, "df2": df2}
    executor = CodeExecutor()
    code = """
merged = pd.merge(df1, df2, on='id')
# Both tables had 'Trạng thái', but code accesses 'Trạng thái' directly
result = merged[merged['Trạng thái'] == 'Hoàn thành']['Doanh số'].sum()
answer_type = 'number'
"""
    res = executor.execute(code, tables, as_of_date="2026-05-01", extract_code=False)
    assert res.success is True
    assert res.result == 300


def test_non_leap_year_feb29_clamping():
    code = "date_val = pd.Timestamp('2025-02-29')"
    patched, _ = patch_code_and_tables_for_rulebook(code, "Doanh thu ngày 29/02/2025?", {})
    assert "2025-02-28" in patched
    assert "2025-02-29" not in patched


def test_code_executor_list_items_unpacking():
    executor = CodeExecutor()
    code = """
monthly_rev = {'2026-01': 100000, '2026-02': 200000, '2026-03': 300000}
result = list(monthly_rev.items())
answer_type = 'list'
"""
    res = executor.execute(code, {}, as_of_date="2026-05-01", extract_code=False)
    assert res.success is True
    assert res.result == [100000, 200000, 300000]


def test_snapshot_table_exempt_from_single_day_penalty():
    gate = ConfidenceGate(min_confidence=0.35)
    ans = {"type": "number", "answer": 94}
    q_info = QuestionInfo(
        original="Hôm nay cửa hàng còn bao nhiêu sản phẩm trong kho?",
        expected_type="number",
        confidence=0.9,
    )
    code = "result = BaoCaoTonKho['Số lượng'].sum()"
    res = gate.evaluate(
        ans,
        question=q_info.original,
        question_info=q_info,
        code=code,
    )
    assert res["type"] == "number"
    assert res["answer"] == 94
