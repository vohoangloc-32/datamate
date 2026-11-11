"""
test_phase7_architecture.py — Comprehensive tests for Phase 7 Rulebook §1–§10 architectural upgrades.
"""

import pandas as pd
import pytest

from src.modules.accounting_cleaner import (
    _PURE_REVENUE_RE,
    _MONETARY_AVG_RE,
    question_is_pure_revenue,
    question_is_monetary_average,
    prepare_tables_for_question,
    patch_code_and_tables_for_rulebook,
    round_half_up_vnd,
    normalize_text,
)
from src.modules.answer_formatter import format_answer
from src.modules.confidence_gate import ConfidenceGate
from src.modules.data_loader import _bridge_relational_tables, load_business_data
from src.modules.question_understanding import QuestionInfo
from src.modules.rulebook_interpreter import Rulebook


def test_pure_revenue_regex_intervening_words():
    # biz_124 style question with "tổng cộng bao nhiêu tiền"
    q1 = "Tháng 1/2025 cửa hàng bán đc tổng cộng bao nhiêu tiền?"
    assert bool(_PURE_REVENUE_RE.search(normalize_text(q1))) is True
    assert question_is_pure_revenue(q1) is True

    # Standard pure revenue
    q2 = "Tổng tiền bán hàng tháng 5/2026 là bao nhiêu?"
    assert bool(_PURE_REVENUE_RE.search(normalize_text(q2))) is True
    assert question_is_pure_revenue(q2) is True

    # Pre-discount column prioritization over Tong tien
    df = pd.DataFrame({
        "Mã đơn hàng": ["HD1", "HD2"],
        "Tiền hàng trước giảm": [100000, 200000],
        "Tong tien": [90000, 180000],
    })
    tables = {"BanHang_HoaDon": df}
    code = "result = BanHang_HoaDon['Tong tien'].sum()"
    patched, _ = patch_code_and_tables_for_rulebook(code, q1, tables)
    assert "Tiền hàng trước giảm" in patched


def test_monetary_average_detection_and_rounding():
    # Monetary average detection
    assert question_is_monetary_average("Giá trị trung bình một đơn hàng là bao nhiêu?") is True
    assert question_is_monetary_average("Mỗi đợt khách mua trung bình bao nhiêu tiền?") is True
    assert question_is_monetary_average("Đơn giá nhập bình quân của mặt hàng X là bao nhiêu?") is True
    assert question_is_monetary_average("Số lượng sản phẩm trung bình mỗi đơn?") is False

    # Integer rounding for monetary averages
    res1 = format_answer(203470.21, answer_type="number", question="Giá trị trung bình mỗi đơn?")
    assert res1["answer"] == 203470
    assert isinstance(res1["answer"], int)

    res2 = format_answer(395367.8, answer_type="number", question="Đơn giá nhập bình quân?")
    assert res2["answer"] == 395368
    assert isinstance(res2["answer"], int)


def test_weighted_average_import_price_rewriting():
    df_nhap = pd.DataFrame({
        "Mã hàng": ["SP01", "SP01"],
        "Số lượng": [10, 20],
        "Đơn giá": [100000, 120000],
    })
    tables = {"ChiTietNhap": df_nhap}
    # Naive unweighted mean code
    code = "result = ChiTietNhap[ChiTietNhap['Mã hàng'] == 'SP01']['Đơn giá'].mean()"
    question = "Đơn giá nhập bình quân của mặt hàng SP01 là bao nhiêu?"
    patched, _ = patch_code_and_tables_for_rulebook(code, question, tables)

    # Must rewrite to weighted average
    assert "sum()" in patched
    assert ".mean()" not in patched


def test_delivery_trip_deduplication_vlxd():
    df_trips = pd.DataFrame({
        "Mã đơn hàng": ["DH01", "DH01", "DH02", "DH02"],
        "Đợt giao": [1, 1, 1, 2],
        "Tên hàng": ["Xi măng", "Cát", "Gạch", "Đá"],
    })
    tables = {"GiaoHang": df_trips}
    code = "result = GiaoHang.shape[0]"
    question = "Tổng cộng có bao nhiêu lượt giao hàng?"
    patched, _ = patch_code_and_tables_for_rulebook(code, question, tables)
    assert "drop_duplicates" in patched


def test_relational_bridging_combined_tables_and_dimensions():
    # Header table combined across months
    df_header = pd.DataFrame({
        "Mã hóa đơn": ["HD01", "HD02"],
        "Ngày bán": pd.to_datetime(["2025-01-10", "2025-02-15"]),
    })
    # Detail table combined
    df_detail = pd.DataFrame({
        "Mã hóa đơn": ["HD01", "HD02"],
        "Mã SKU": ["TH01", "TH02"],
        "Số lượng": [2, 5],
    })
    # Drug catalog
    df_catalog = pd.DataFrame({
        "Mã SKU": ["TH01", "TH02"],
        "Tên thuốc": ["Panadol", "Amoxicillin"],
        "Hoạt chất": ["Paracetamol", "Amoxicillin"],
        "Thuốc kê đơn": [False, True],
    })

    tables = {
        "HoaDon__monthly_combined": df_header,
        "ChiTiet__monthly_combined": df_detail,
        "DanhMucThuoc": df_catalog,
    }

    bridged = _bridge_relational_tables(tables)
    detail_bridged = bridged["ChiTiet__monthly_combined"]

    # Date propagated into combined detail table
    assert "Ngày bán" in detail_bridged.columns
    # Product dimensions propagated into combined detail table
    assert "Tên thuốc" in detail_bridged.columns
    assert "Hoạt chất" in detail_bridged.columns
    assert "Thuốc kê đơn" in detail_bridged.columns
    assert detail_bridged["Thuốc kê đơn"].iloc[1] == True


def test_customer_dimension_bridging_with_mismatched_ids():
    df_cust = pd.DataFrame({
        "Mã khách hàng": ["KH01", "KH02"],
        "Tên khách hàng": ["Nguyễn Văn A", "Trần Thị B"],
        "Nhóm khách hàng": ["VIP", "Sỉ"],
    })
    df_invoice = pd.DataFrame({
        "Mã KH": ["KH01", "KH02"],
        "Mã hóa đơn": ["HD1", "HD2"],
        "Tổng tiền": [500000, 1000000],
    })
    tables = {
        "DanhSachKhachHang": df_cust,
        "DanhSachHoaDon": df_invoice,
    }
    bridged = _bridge_relational_tables(tables)
    assert "Tên khách hàng" in bridged["DanhSachHoaDon"].columns
    assert "Mã khách hàng" in bridged["DanhSachHoaDon"].columns
    assert bridged["DanhSachHoaDon"]["Tên khách hàng"].iloc[0] == "Nguyễn Văn A"


def test_confidence_gate_domain_plausibility_unsold_and_debt():
    gate = ConfidenceGate(min_confidence=0.35)

    # 1. Unsold goods count > 30 items
    ans_unsold_bad = {"type": "number", "answer": 150}
    q_unsold = "Có bao nhiêu mặt hàng ế không bán được trong năm qua?"
    q_info = QuestionInfo(original=q_unsold, expected_type="number", confidence=0.9)
    res_unsold = gate.evaluate(ans_unsold_bad, question=q_unsold, question_info=q_info)
    assert res_unsold["type"] == "abstain"

    # Plausible unsold goods count <= 30 items
    ans_unsold_good = {"type": "number", "answer": 2}
    res_unsold_ok = gate.evaluate(ans_unsold_good, question=q_unsold, question_info=q_info)
    assert res_unsold_ok["type"] == "number"
    assert res_unsold_ok["answer"] == 2

    # 2. Implausible debt payment > 100M VND
    ans_debt_bad = {"type": "number", "answer": 850_000_000}
    q_debt = "Khách hàng Nguyễn Văn A đã trả bớt nợ bao nhiêu tiền?"
    q_info_debt = QuestionInfo(original=q_debt, expected_type="number", confidence=0.9)
    res_debt = gate.evaluate(ans_debt_bad, question=q_debt, question_info=q_info_debt)
    assert res_debt["type"] == "abstain"


def test_readme_unit_conversion_table_loading():
    tables = load_business_data("data/train/data/biz_101", use_cache=False)
    assert "DanhSachQuyDoi" in tables
    assert len(tables["DanhSachQuyDoi"]) == 77
    assert "Tên sản phẩm" in tables["DanhSachQuyDoi"].columns
    assert "Tỉ lệ quy đổi" in tables["DanhSachQuyDoi"].columns
    assert tables["DanhSachQuyDoi"]["Tỉ lệ quy đổi"].iloc[0] == 6
