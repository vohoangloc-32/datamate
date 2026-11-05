"""
Tests for accounting_cleaner module and its integration with data_loader,
code_executor, and confidence_gate.
"""

import pandas as pd
import pytest

from src.modules.accounting_cleaner import (
    classify_status,
    dedup_order_table_rows,
    detect_status_column,
    find_order_id_column,
    patch_code_and_tables_for_rulebook,
    prepare_tables_for_question,
    question_is_average_order_value,
    question_is_customer_ranking,
    question_is_pure_revenue,
    question_needs_all_statuses,
    remove_subheader_and_subtotal_rows,
    round_half_up_vnd,
)
from src.modules.code_executor import CodeExecutor
from src.modules.confidence_gate import ConfidenceGate
from src.modules.data_loader import load_business_data
from src.modules.question_understanding import QuestionInfo


def test_classify_status_four_canonical_classes_without_hoan_thanh_bug():
    # Completed variants (must never be confused with 'hoàn' in 'Hoàn trả')
    for val in [
        "Hoàn thành",
        "Đã hoàn thành",
        "xong",
        "đã xong",
        "Thành công",
        "đã lấy",
        "Đã giao",
        "ok",
        "completed",
    ]:
        assert classify_status(val) == "completed", f"Failed on {val!r}"

    # Return variants
    for val in [
        "Đã hoàn hàng",
        "Hoàn trả",
        "Đổi trả",
        "Trả hàng",
        "trả hàng",
        "Đã trả hàng",
        "trả lại",
        "hàng trả về",
        "returned",
    ]:
        assert classify_status(val) == "return", f"Failed on {val!r}"

    # Cancel variants
    for val in [
        "Đã hủy",
        "Đã huỷ",
        "hủy",
        "Hủy",
        "hủy đơn",
        "Huỷ đơn",
        "Đã hủy đơn",
        "khách hủy",
        "Không giao được",
        "cancel",
        "cancelled",
    ]:
        assert classify_status(val) == "cancel", f"Failed on {val!r}"

    # Shipping / in-progress variants (valid for revenue per Rulebook §1)
    for val in [
        "Đang giao",
        "Đang giao hàng",
        "Đang giao dịch",
        "Đang vận chuyển",
        "Đang xử lý",
        "đang gửi",
        "Chờ giao",
        "shipping",
    ]:
        assert classify_status(val) == "shipping", f"Failed on {val!r}"


def test_subtotal_removal_preserves_surname_tong_and_product_cong_tac():
    df = pd.DataFrame(
        [
            ["2025-02-01", "ORD001", "Tống Khánh Quang", "Công tắc 1 chiều Panasonic Wide kèm mặt", 2, 35000, 70000, "Đã hoàn thành"],
            ["2025-02-01", "ORD002", "Tống Nam Minh", "Cộng sản phẩm đặc biệt", 1, 50000, 50000, "Đã hoàn thành"],
            [None, None, None, "TỔNG NGÀY 01/02", None, None, 120000, None],
            [None, None, "Cộng ngày 01/02/2025", None, None, None, 120000, None],
            [None, None, "Tổng tháng 02/2025", None, None, None, 120000, None],
        ],
        columns=[
            "Ngày",
            "Mã đơn hàng",
            "Tên khách hàng",
            "Tên hàng hóa",
            "Số lượng",
            "Đơn giá",
            "Thành tiền",
            "Trạng thái",
        ],
    )
    cleaned = remove_subheader_and_subtotal_rows(df)
    assert len(cleaned) == 2
    assert cleaned["Tên khách hàng"].tolist() == ["Tống Khánh Quang", "Tống Nam Minh"]
    assert "Công tắc 1 chiều Panasonic Wide kèm mặt" in cleaned["Tên hàng hóa"].tolist()


def test_data_loader_removes_subtotal_before_ffill(tmp_path):
    biz_dir = tmp_path / "biz_subtotal_ffill"
    biz_dir.mkdir()
    pd.DataFrame(
        [
            ["2025-02-01", "ORD001", "Chân gà nướng", "đĩa", 2, 85000, 170000, "Đã hoàn thành"],
            [None, None, "TỔNG NGÀY 01/02", None, None, None, 170000, None],
            ["2025-02-02", "ORD002", "Lẩu Thái", "nồi", 1, 189000, 189000, "Đã hoàn thành"],
        ],
        columns=[
            "Ngày",
            "Mã đơn hàng",
            "Tên hàng hóa",
            "Đơn vị tính",
            "Số lượng",
            "Đơn giá",
            "Thành tiền",
            "Trạng thái",
        ],
    ).to_excel(biz_dir / "sales.xlsx", index=False)

    tables = load_business_data(str(biz_dir), use_cache=False)
    sales = tables["sales"]
    assert len(sales) == 2
    assert sales["Mã đơn hàng"].tolist() == ["ORD001", "ORD002"]
    assert sales["Thành tiền"].sum() == 359000


def test_dedup_order_table_vs_non_order_table():
    # Order-keyed table: exact duplicates within same order are export bugs (Rulebook §8 line 111)
    order_df = pd.DataFrame(
        [
            ["ORD001", "Nước ép cam (M)", 2, 40000],
            ["ORD001", "Nước ép cam (M)", 2, 40000],
            ["ORD002", "Nước ép cam (M)", 2, 40000],
        ],
        columns=["Mã hóa đơn", "Hàng", "SL", "Tiền"],
    )
    assert len(dedup_order_table_rows(order_df)) == 2

    # Non-order table: identical rows represent separate sales (Rulebook §8 line 112)
    non_order_df = pd.DataFrame(
        [
            ["2025-03-01", "Nước ép cam (M)", 2, 40000],
            ["2025-03-01", "Nước ép cam (M)", 2, 40000],
        ],
        columns=["ngay_ban", "Nội dung", "Số lượng", "Tổng tiền"],
    )
    assert len(dedup_order_table_rows(non_order_df)) == 2


def test_metric_aware_status_filtering_and_detail_propagation():
    header = pd.DataFrame(
        [
            ["ORD01", "Hoàn thành", 100000],
            ["ORD02", "Đã hủy", 50000],
            ["ORD03", "Đã trả hàng", 30000],
            ["ORD04", "Đang giao hàng", 20000],
        ],
        columns=["Mã hóa đơn", "Trạng thái", "Tổng tiền hàng"],
    )
    detail = pd.DataFrame(
        [
            ["ORD01", "SP1", 100000],
            ["ORD02", "SP2", 50000],
            ["ORD03", "SP3", 30000],
            ["ORD04", "SP1", 20000],
        ],
        columns=["Mã hóa đơn", "Mã SP", "Thành tiền"],
    )
    raw = {"DanhSachHoaDon": header, "DanhSachHoaDonChiTiet": detail}

    # 1. Revenue question -> filters out ORD02 and ORD03 from BOTH header and detail, keeps ORD04 ('Đang giao hàng')
    rev_tables = prepare_tables_for_question(raw, "Tổng doanh thu tháng này là bao nhiêu?")
    assert rev_tables["DanhSachHoaDon"]["Mã hóa đơn"].tolist() == ["ORD01", "ORD04"]
    assert rev_tables["DanhSachHoaDonChiTiet"]["Mã hóa đơn"].tolist() == ["ORD01", "ORD04"]
    assert rev_tables["DanhSachHoaDonChiTiet"]["Thành tiền"].sum() == 120000

    # 2. Cancellation question -> keeps all 4 orders
    cancel_tables = prepare_tables_for_question(raw, "Cửa hàng có mấy đơn ở trạng thái huỷ?")
    assert len(cancel_tables["DanhSachHoaDon"]) == 4

    # 3. Debt question (§4) -> keeps all 4 orders
    debt_tables = prepare_tables_for_question(raw, "Tổng công nợ còn lại tính đến hôm nay là bao nhiêu?")
    assert len(debt_tables["DanhSachHoaDon"]) == 4


def test_walkin_customer_filter_preserves_nhom_khach_hang_le():
    invoices = pd.DataFrame(
        [
            ["ORD01", "KH01", "Khách lẻ", 500000, "Hoàn thành"],
            ["ORD02", "KH02", "Vương Phúc Dương", 300000, "Hoàn thành"],
            ["ORD03", "KH03", "lẻ", 400000, "Hoàn thành"],
            ["ORD04", "KH04", "Triệu Cường Thanh", 250000, "Hoàn thành"],
        ],
        columns=["Mã hóa đơn", "Mã KH", "Khách hàng", "Tổng tiền hàng", "Trạng thái"],
    )
    customers = pd.DataFrame(
        [
            ["KH02", "Vương Phúc Dương", "lẻ"],
            ["KH04", "Triệu Cường Thanh", "lẻ"],
        ],
        columns=["Mã khách hàng", "Tên khách hàng", "Nhóm khách hàng"],
    )
    raw = {"DanhSachHoaDon": invoices, "DanhSachKhachHang": customers}

    prepared = prepare_tables_for_question(raw, "Top 2 khách hàng mua nhiều tiền nhất là ai?")
    # Walk-in names ('Khách lẻ', 'lẻ') removed from invoice customer name column
    assert prepared["DanhSachHoaDon"]["Khách hàng"].tolist() == [
        "Vương Phúc Dương",
        "Triệu Cường Thanh",
    ]
    # Registered customers in DanhSachKhachHang with Nhóm khách hàng == 'lẻ' MUST NOT be dropped
    assert len(prepared["DanhSachKhachHang"]) == 2


def test_code_executor_rulebook_revenue_column_and_aov_rounding():
    df = pd.DataFrame(
        [
            ["ORD01", "xong", 100001, 90000],
            ["ORD02", "Hoàn thành", 200000, 180000],
        ],
        columns=["Mã hóa đơn", "Trạng thái", "Tổng tiền hàng", "Tổng sau giảm giá"],
    )
    tables = {"DanhSachHoaDon": df}
    executor = CodeExecutor()

    # 1. Pure revenue question where LLM mistakenly filtered == 'Hoàn thành' and summed 'Tổng sau giảm giá'
    code = """
valid = DanhSachHoaDon[DanhSachHoaDon['Trạng thái'] == 'Hoàn thành']
result = valid['Tổng sau giảm giá'].sum()
answer_type = 'number'
"""
    res = executor.execute(
        code,
        tables,
        as_of_date="2026-06-01",
        extract_code=False,
        question="Cho tôi biết tổng doanh thu là bao nhiêu?",
    )
    assert res.success
    assert res.result == 300001

    # 2. Average order value question rounds half up to integer VND (150000.5 -> 150001)
    aov_code = """
result = DanhSachHoaDon['Tổng tiền hàng'].mean()
answer_type = 'number'
"""
    res_aov = executor.execute(
        aov_code,
        tables,
        as_of_date="2026-06-01",
        extract_code=False,
        question="Trung bình mỗi đơn mang về bao nhiêu doanh thu?",
    )
    assert res_aov.success
    assert res_aov.result == 150001


def test_confidence_gate_abstains_on_high_risk_percent_and_retries():
    gate = ConfidenceGate(min_confidence=0.45)
    q_info = QuestionInfo(expected_type="percent", confidence=0.8)

    # Degenerate -100% or 0% -> abstain
    res_deg = gate.evaluate(
        {"type": "percent", "answer": -100.0},
        question_info=q_info,
        execution_retries=0,
        question="Doanh thu tháng 5 thay đổi bao nhiêu %?",
    )
    assert res_deg["type"] == "abstain"

    # Non-revenue percent question -> abstain
    res_non_rev = gate.evaluate(
        {"type": "percent", "answer": 1.35},
        question_info=q_info,
        execution_retries=0,
        question="Tháng 5/2025 tỉ lệ đơn bị huỷ là bao nhiêu %?",
    )
    assert res_non_rev["type"] == "abstain"

    # Clean revenue percent question with 0 retries -> accepted
    res_rev = gate.evaluate(
        {"type": "percent", "answer": 17.62},
        question_info=q_info,
        execution_retries=0,
        question="Nhóm sơn đóng góp bao nhiêu phần trăm doanh thu cả cửa hàng?",
    )
    assert res_rev["type"] == "percent"
    assert res_rev["answer"] == 17.62
