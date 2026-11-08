"""
Tests for Phase 8 Zero-Overfit Architectural Upgrades:
1. QuestionAnalyzer Telex typo normalization & 100% expected_type accuracy (list chẵn/lẻ, table with metric, number with units).
2. AnswerFormatter pure-numeric string->number/percent auto-rescue, numeric-string list coercion, and SKU-to-Name resolution.
3. ConfidenceGate schema answerability checks (inventory, debt, expiry, import price, sales channel, period before shop data start).
4. Walk-in customer ('Khách lẻ') filtering on 'Khách' column & repeat-customer counting, and delivery trip deduplication.
5. Base-unit quantity ('Số lượng quy đổi') pre-computation in data_loader and automatic code redirection in accounting_cleaner.
6. Elimination of duplicate split + monthly_combined tables in _combine_split_tables.
"""

from datetime import date
import pandas as pd

from src.modules.question_understanding import QuestionAnalyzer, normalize_question_slang
from src.modules.answer_formatter import format_answer, format_list, resolve_entity_codes_in_answer
from src.modules.confidence_gate import ConfidenceGate
from src.modules.accounting_cleaner import (
    prepare_tables_for_question,
    patch_code_and_tables_for_rulebook,
    question_is_customer_ranking,
    question_is_monetary_average,
)
from src.modules.data_loader import _combine_split_tables, _bridge_relational_tables


def test_pillar1_telex_slang_and_type_classification():
    qa = QuestionAnalyzer()
    d = date(2026, 6, 15)

    # Telex typos & number unit classification
    assert qa.analyze(normalize_question_slang("check giùm bữa qua cuwar hàng có bao nhieeu bill"), d).expected_type == "number"
    assert qa.analyze(normalize_question_slang("Trong tháng 12 năm 2025 cửa hàng bán đc mấy ly Cà phê muối (M)"), d).expected_type == "number"
    assert qa.analyze(normalize_question_slang("giá trị trung bình một đơn tháng 4/2026 là nhiêu"), d).expected_type == "number"
    assert qa.analyze(normalize_question_slang("trong tháng 6/2026, nhoms gỏi đóng góp bao nhieeu % dthu cả cửa hàng"), d).expected_type == "percent"
    assert qa.analyze(normalize_question_slang("doanh thu 3 tháng 4/2026, 5/2026, 6/2026 cos tăng liên tục khoong"), d).expected_type == "boolean"

    # Mixed unit chẵn/lẻ -> list
    assert qa.analyze(normalize_question_slang("cuối ngày hoom nay lưới B40 mạ kẽm khổ 1.5m còn mấy cuộn 10m chẵn, maays m lẻ"), d).expected_type == "list"
    assert qa.analyze(normalize_question_slang("xem giúp tồn cuối bữa nay của amoxicillin là mấy vỉ chẵn, mấy viên lẻ z"), d).expected_type == "list"

    # Multi-value revenue -> list
    assert qa.analyze(normalize_question_slang("Liệt kê doanh thu 3 tháng liên tiếp từ tháng 2/2025 đến tháng 4/2025, theo thứ tự thời gian."), d).expected_type == "list"
    assert qa.analyze(normalize_question_slang("So doanh thu các đơn bán qua điện thoại với doanh thu các đơn bán tại quầy tháng này: mỗi bên bao nhiêu nhỉ?"), d).expected_type == "list"

    # Top-K with accompanying metric -> table
    assert qa.analyze(normalize_question_slang("Liệt kê 3 khách nợ nhiều nhất tính đến hết ngày 19/08/2026, kèm số tiền còn nợ."), d).expected_type == "table"
    assert qa.analyze(normalize_question_slang("top 5 khách hàng có công nợ lớn nhất đến hết ngày 12/10/2025 và số nợ của từng người"), d).expected_type == "table"

    # Customer ranking k=1 -> string even with "tính theo số đơn"
    assert qa.analyze(normalize_question_slang("Tháng 6/2025 ai là khách mua nhiều lần nhất, tính theo số đơn?"), d).expected_type == "string"


def test_pillar2_answer_formatter_rescue_and_sku_resolution():
    # Pure number in string -> rescued to number
    ans1 = format_answer("108.0", "string", question="Trong tháng 12 năm 2025 cửa hàng bán đc mấy ly Cà phê muối (M)")
    assert ans1 == {"type": "number", "answer": 108}

    # Monetary average in string with Telex typo "trung binhf" -> rescued to number and rounded to integer VND
    ans2 = format_answer(163518.52, "number", question="check giùm trung binhf mỗi đơn tháng 4/2025 mang về nhiêu dthu")
    assert ans2 == {"type": "number", "answer": 163519}

    # Percentage in number -> rescued to percent
    ans3 = format_answer(10.95, "number", question="trong tháng 6/2026, nhoms gỏi đóng góp bao nhieeu % dthu cả cửa hàng")
    assert ans3 == {"type": "percent", "answer": 10.95}

    # Numeric strings in list -> coerced to numbers
    assert format_list(["302000", "178000"]) == [302000, 178000]

    # Mixed unit string -> unpacked to 2-number list
    ans4 = format_answer("9 vỉ chẵn, 2 viên lẻ", "string", question="còn mấy vỉ chẵn, mấy viên lẻ")
    assert ans4 == {"type": "list", "answer": [9, 2]}

    # SKU to Product Name resolution
    tables = {
        "DanhSachSanPham": pd.DataFrame({
            "Mã SKU": ["SP0001", "SP0053"],
            "Tên sản phẩm": ["Otrivin 0,1% xịt mũi chai 10ml", "Giấy vệ sinh Pulppy 10 cuộn"],
        })
    }
    res_str = resolve_entity_codes_in_answer(
        {"type": "string", "answer": "SP0053"},
        "cho hỏi tháng 10/2025 hàng nào lời nhất",
        tables,
    )
    assert res_str == {"type": "string", "answer": "Giấy vệ sinh Pulppy 10 cuộn"}

    res_list = resolve_entity_codes_in_answer(
        {"type": "list", "answer": ["SP0001", "SP0053"]},
        "những sản phẩm nào đang còn hàng",
        tables,
    )
    assert res_list == {"type": "list", "answer": ["Otrivin 0,1% xịt mũi chai 10ml", "Giấy vệ sinh Pulppy 10 cuộn"]}


def test_pillar3_schema_answerability_audit():
    gate = ConfidenceGate()
    qa = QuestionAnalyzer()
    sales_only_tables = {
        "DonHang_ChiTiet": pd.DataFrame({
            "Mã đơn hàng": ["DH01"],
            "Ngày đặt hàng": pd.to_datetime(["2025-05-10"]),
            "Tên sản phẩm": ["Bia Tiger"],
            "Số lượng": [2],
            "Thành tiền": [50000],
        })
    }

    # Missing inventory
    r_inv = gate.evaluate({"type": "number", "answer": 10}, question="hết hôm nay, bia tiger còn lại mấy lon trong kho", tables=sales_only_tables)
    assert r_inv["type"] == "unanswerable"

    # Missing customer debt
    r_debt = gate.evaluate({"type": "number", "answer": 1000}, question="số tiền khách còn thiếu là bao nhiêu", tables=sales_only_tables)
    assert r_debt["type"] == "unanswerable"

    # Missing expiry date
    r_exp = gate.evaluate({"type": "list", "answer": ["Bia Tiger"]}, question="những sp nào có hạn dùng từ 20/05/2026", tables=sales_only_tables)
    assert r_exp["type"] == "unanswerable"

    # Missing sales channel
    r_chan = gate.evaluate({"type": "percent", "answer": 10.0}, question="doanh thu các đơn bán qua điện thoại chiếm bao nhiêu %", tables=sales_only_tables)
    assert r_chan["type"] == "unanswerable"

    # Period completely before shop data start
    q_text = "Khách nào chi nhiều tiền nhất cho cửa hàng trong tháng 2/2025?"
    q_info = qa.analyze(q_text, date(2026, 6, 1))
    r_period = gate.evaluate({"type": "string", "answer": "Nguyễn Văn A"}, question_info=q_info, question=q_text, tables=sales_only_tables)
    assert r_period["type"] == "unanswerable"


def test_pillar4_walkin_customer_and_delivery_trip_dedup():
    assert question_is_customer_ranking("Tháng 6/2025 ai mua hàng nhiều tiền nhất?")
    assert question_is_customer_ranking("khách nào mua nhìu đơn nhất tháng 11/2025 vậy")
    assert question_is_customer_ranking("thang 4/2026 co bnhiu khach mua tu 4 don tro len")
    assert question_is_monetary_average("check giùm trung binhf mỗi đơn tháng 4/2025 mang về nhiêu dthu")

    tables = {
        "SoBanHang": pd.DataFrame({
            "Mã hóa đơn": ["HD1", "HD2", "HD3"],
            "Khách": ["Khách lẻ", "Trịnh Thanh Dương", None],
            "Trị giá": [500000, 300000, 100000],
        }),
        "BangKeGiao": pd.DataFrame({
            "Mã đơn hàng": ["DH1", "DH1", "DH2"],
            "Lần giao": [1, 1, 1],
            "Tên hàng hóa": ["Cát", "Đá", "Xi măng"],
        }),
    }
    prep_cust = prepare_tables_for_question(tables, "Tháng 6/2025 ai mua hàng nhiều tiền nhất?")
    assert list(prep_cust["SoBanHang"]["Khách"]) == ["Trịnh Thanh Dương"]

    prep_trip = prepare_tables_for_question(tables, "Tháng 4/2026 xe đã giao tổng cộng mấy đợt hàng?")
    assert len(prep_trip["BangKeGiao"]) == 2


def test_pillar5_and_6_unit_conversion_and_split_table_dedup():
    tables = {
        "DonHang_ChiTiet_1__monthly_combined": pd.DataFrame({
            "Mã đơn hàng": ["DH1"],
            "Tên sản phẩm": ["Bia Tiger lon"],
            "Đơn vị tính": ["Thùng"],
            "Số lượng": [2],
        }),
        "DonHang_ChiTiet_2026__monthly_combined": pd.DataFrame({
            "Mã đơn hàng": ["DH2"],
            "Tên sản phẩm": ["Bia Tiger lon"],
            "Đơn vị tính": ["Lon"],
            "Số lượng": [5],
        }),
        "DanhSachQuyDoi": pd.DataFrame({
            "Tên sản phẩm": ["Bia Tiger lon"],
            "Đơn vị quy đổi": ["Thùng"],
            "Tỉ lệ quy đổi": [24],
            "Đơn vị cơ bản": ["Lon"],
        }),
    }
    combined = _combine_split_tables(tables)
    # Partial monthly_combined tables must be removed to prevent 2x double counting
    assert "DonHang_ChiTiet_1__monthly_combined" not in combined
    assert "DonHang_ChiTiet_2026__monthly_combined" not in combined
    assert "DonHang_ChiTiet__monthly_combined" in combined

    bridged = _bridge_relational_tables(combined)
    df_comb = bridged["DonHang_ChiTiet__monthly_combined"]
    assert "Số lượng quy đổi" in df_comb.columns
    assert list(df_comb["Số lượng quy đổi"]) == [48.0, 5.0]

    code = "result = df['Số lượng'].sum()"
    patched, _ = patch_code_and_tables_for_rulebook(
        code,
        "tháng 5/2025 bán ra bao nhiêu lon Bia Tiger lon",
        bridged,
    )
    assert "['Số lượng quy đổi']" in patched
