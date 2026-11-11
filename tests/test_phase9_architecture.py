"""
Tests for Phase 9 Zero-Overfit Upgrades:
1. format_list with numpy integer/float retains numeric types (no string conversion).
2. resolve_entity_codes_in_answer drops redundant code column from 3-column tables.
3. patch_code_and_tables_for_rulebook replaces inner join on conversion tables with left join.
4. patch_code_and_tables_for_rulebook enforces (M) / (L) regex for size questions per Rulebook §6.
5. patch_code_and_tables_for_rulebook drops cancelled import slips per Rulebook §5 line 63.
"""

import numpy as np
import pandas as pd

from src.modules.answer_formatter import format_list, resolve_entity_codes_in_answer
from src.modules.accounting_cleaner import patch_code_and_tables_for_rulebook


def test_phase9_format_list_numpy_types():
    raw_list = [np.int64(74000), np.int64(37000)]
    res = format_list(raw_list)
    assert res == [74000, 37000]
    assert all(isinstance(x, (int, float)) for x in res)


def test_phase9_table_strip_code_column():
    ans = {
        "type": "table",
        "answer": {
            "columns": ["Customer ID", "Tên khách hàng", "Còn nợ"],
            "rows": [["CUST001", "Nguyễn Văn A", 100000], ["CUST002", "Trần Thị B", 200000]],
        },
    }
    resolved = resolve_entity_codes_in_answer(ans, "Cho bảng 3 khách còn nợ nhiều nhất", {})
    assert resolved["answer"]["columns"] == ["Tên khách hàng", "Còn nợ"]
    assert resolved["answer"]["rows"] == [["Nguyễn Văn A", 100000], ["Trần Thị B", 200000]]


def test_phase9_conversion_table_left_join_patch():
    code = "merged = df.merge(danh_sach_quy_doi, on='Tên sản phẩm')"
    patched, _ = patch_code_and_tables_for_rulebook(code, "bán ra bao nhiêu kg nếp cái hoa vàng", {})
    assert "how='left'" in patched


def test_phase9_size_m_regex_patch():
    code = "m_drinks = df[df['Tên sản phẩm'].str.contains('M')]"
    patched, _ = patch_code_and_tables_for_rulebook(code, "trong tháng 1/2026 số ly size M bán được là bao nhiêu", {})
    assert r"\(M\)\s*$" in patched


def test_phase9_import_slip_cancel_filter_patch():
    tables = {
        "DanhSachNhapHang": pd.DataFrame({
            "Mã nhập hàng": ["PN01", "PN02"],
            "Trạng thái": ["Đã nhập hàng", "Đã hủy"],
            "Cần trả NCC": [1000000, 5000000],
        })
    }
    code = "result = DanhSachNhapHang['Cần trả NCC'].sum()"
    patched, _ = patch_code_and_tables_for_rulebook(code, "Tổng tiền hàng nhập vào tháng 7/2025 là bao nhiêu", tables)
    assert "huy|cancel" in patched
