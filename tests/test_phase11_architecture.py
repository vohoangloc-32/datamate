"""
test_phase11_architecture.py — Unit Tests for Phase 11 Zero-Overfit Upgrades

Covers:
1. Tuple unpacking in format_list (e.g. divmod result [16, 9]).
2. Quantity parsing in _coerce_types for compound units with digits (e.g. '1 hộp 30', '2 bao 25kg').
3. Delivery trips deduplication unaffected by 'bao nhiêu đợt'.
4. Base-unit quantity fallback injection for product ranking.
5. Prompt accounting notes for unsold goods ('mặt hàng bị ế'), debt added/paid, blister inventory, and price change comparisons.
"""

import pandas as pd
import pytest

from src.modules.answer_formatter import format_list
from src.modules.data_loader import _coerce_types
from src.modules.accounting_cleaner import (
    prepare_tables_for_question,
    patch_code_and_tables_for_rulebook,
    format_accounting_context,
)


def test_tuple_in_format_list():
    """Ensure tuple output (e.g. from divmod) is formatted cleanly into a list."""
    assert format_list((16, 9)) == [16, 9]
    assert format_list((8, 6)) == [8, 6]
    assert format_list((10000, 20000)) == [10000, 20000]


def test_coerce_types_compound_quantity_units():
    """Ensure unit strings with digits (e.g. '1 hộp 30', '2 bao 25kg') are parsed to numeric."""
    df = pd.DataFrame({
        "Số lượng sản phẩm": ["1 hộp 30", "2 bao 25kg", "3 gói", 5, 10, "1 chai", "2 lon"],
    })
    coerced = _coerce_types(df)
    assert coerced["Số lượng sản phẩm"].isna().sum() == 0
    assert coerced["Số lượng sản phẩm"].tolist() == [1.0, 2.0, 3.0, 5.0, 10.0, 1.0, 2.0]


def test_delivery_trips_deduplication():
    """Ensure 'bao nhiêu đợt' is not blocked by bag units and dedup is applied."""
    df = pd.DataFrame({
        "Mã bill": ["B01", "B01", "B02"],
        "Lượt giao": [1, 1, 1],
        "Ngày giao": ["2026-04-01", "2026-04-01", "2026-04-02"],
    })
    tables = {"SoGiaoHang": df}
    prep = prepare_tables_for_question(tables, "tháng 4/2026 cửa hàng đã cho giao bao nhiêu đợt hàng cho khách")
    assert len(prep["SoGiaoHang"]) == 2


def test_product_ranking_sl_quy_doi_fallback():
    """Ensure product ranking code patch works cleanly and injects 'Số lượng quy đổi' fallback."""
    df = pd.DataFrame({
        "Tên sản phẩm": ["SP_A", "SP_B"],
        "Số lượng hàng": [10, 20],
        "Thành tiền": [100000, 200000],
    })
    tables = {"DonHang": df}
    code = "result = DonHang.groupby('Tên sản phẩm')['Thành tiền'].sum().nlargest(1).index.tolist()"
    patched_code, exec_tables = patch_code_and_tables_for_rulebook(code, "hàng nào bán chạy nhất", tables)
    assert "Số lượng quy đổi" in patched_code
    assert "Số lượng quy đổi" in exec_tables["DonHang"].columns
    assert exec_tables["DonHang"]["Số lượng quy đổi"].tolist() == [10.0, 20.0]


def test_accounting_notes_triggers():
    """Verify Rulebook prompt guidance triggers for unsold goods, debt, blister inventory, and price changes."""
    df = pd.DataFrame({"Mã đơn": ["D1"], "Ngày": ["2025-01-01"]})
    tables = {"don_hang": df}

    # 1. Unsold goods with 'mặt hàng bị ế'
    notes1 = format_accounting_context(tables, "Trong tuần qua mặt hàng bị ế là bao nhiêu")
    assert notes1 is not None and "Hàng ế trong kỳ" in notes1

    # 2. Debt added and paid
    notes2 = format_accounting_context(tables, "khách nợ thêm bao nhiêu và trả nợ được bao nhiêu")
    assert notes2 is not None and "Công nợ phát sinh" in notes2

    # 3. Blister / Pill even/odd inventory
    notes3 = format_accounting_context(tables, "còn bao nhiêu vỉ chẵn, lẻ bao nhiêu viên")
    assert notes3 is not None and "Tồn kho vỉ chẵn" in notes3

    # 4. Price change comparison
    notes4 = format_accounting_context(tables, "sản phẩm A tăng giá vào ngày 10/05/2025, 7 ngày kể từ ngày đó so với 7 ngày ngay trước thay đổi bao nhiêu %")
    assert notes4 is not None and "So sánh lượng bán" in notes4
