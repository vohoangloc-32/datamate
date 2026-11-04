"""
Tests for question_understanding module.
"""

import pytest
from datetime import date
from src.modules.question_understanding import QuestionAnalyzer, QuestionInfo


@pytest.fixture
def analyzer():
    return QuestionAnalyzer()


@pytest.fixture
def as_of_date():
    return date(2024, 5, 18)


def test_classify_number(analyzer, as_of_date):
    q = "Có bao nhiêu đơn hàng đã hoàn thành trong tháng này?"
    info = analyzer.analyze(q, as_of_date)
    assert info.expected_type == "number"
    assert info.aggregation in ("count", "sum")


def test_classify_percent(analyzer, as_of_date):
    q = "Doanh thu tuần này tăng bao nhiêu phần trăm so với tuần trước?"
    info = analyzer.analyze(q, as_of_date)
    assert info.expected_type == "percent"
    assert info.is_comparison is True


def test_classify_string_and_ranking(analyzer, as_of_date):
    q = "Sản phẩm nào bán chạy nhất trong quý 1?"
    info = analyzer.analyze(q, as_of_date)
    assert info.expected_type == "string"
    assert info.ranking_direction == "desc"
    assert info.ranking_k == 1
    assert info.ranking_criterion == "quantity"


def test_classify_date(analyzer, as_of_date):
    q = "Ngày nào có số lượng đơn hàng cao nhất trong năm 2023?"
    info = analyzer.analyze(q, as_of_date)
    assert info.expected_type == "date"


def test_classify_boolean(analyzer, as_of_date):
    q = "Có khách hàng nào mua hàng quá 10 lần trong tháng không?"
    info = analyzer.analyze(q, as_of_date)
    assert info.expected_type == "boolean"


def test_classify_list(analyzer, as_of_date):
    q = "Liệt kê danh sách top 3 chi nhánh có doanh thu cao nhất."
    info = analyzer.analyze(q, as_of_date)
    assert info.expected_type == "list"
    assert info.ranking_k == 3


@pytest.mark.parametrize(
    ("question", "expected_type"),
    [
        ("e bua nay cua hang ban duoc bao nhieu tien", "number"),
        ("dthu trung bình treen một đơn tháng 7/2025 là bnhiu", "number"),
        ("so don bi huy trong thang 7 nam 2026 la bao nhieu", "number"),
        ("Tháng 4 năm 2025, 5 khách nào chi tiêu nhiều nhất?", "list"),
        ("thang 8/2026 ti le don bi huy la bn %", "percent"),
        ("Doanh thu 3 tháng có tăng liên tục ko?", "boolean"),
        ("ngay nao trong thang 9/2025 doanh thu cao nhat", "date"),
        ("tháng 1/2026 ai mua hàng nhìu tiền nhất", "string"),
        ("thang 8/2026 khach nao mua nhieu don nhat", "string"),
        ("Xem giúp ngày 01/05/2026 cửa hàng bán Povidine 10% chai 90ml giá bao nhiêu một chai?", "number"),
        ("Cho tôi biết doanh thu tháng 8/2025.", "number"),
        ("sản phẩm nào có doanh thu cao nhất?", "string"),
        ("Doanh thu các đơn Zalo OA và tại quầy lần lượt là bao nhiêu?", "list"),
        ("doanh thu tháng 1/2026 tăng hay giảm nhiêu % so với tháng trước", "percent"),
        ("Doanh thu 3 tháng liền nhau, mỗi tháng là bao nhiêu?", "list"),
        ("cho bảng 3 khách còn nợ nhiều nhất, kèm số tiền còn nợ", "table"),
        ("tồn kho được mấy lốc chẵn và mấy lọ lẻ", "list"),
        ("Cho tôi 3 khách mua nhiều tiền nhất tháng 4/2026.", "list"),
        ("5 best seller tháng 2/2026 là những mặt hàng nào", "list"),
    ],
)
def test_classifies_common_unaccented_and_informal_questions(
    analyzer,
    as_of_date,
    question,
    expected_type,
):
    assert analyzer.analyze(question, as_of_date).expected_type == expected_type


def test_classify_table(analyzer, as_of_date):
    q = "Bảng tổng hợp doanh thu và số đơn hàng theo từng tháng năm 2023."
    info = analyzer.analyze(q, as_of_date)
    assert info.expected_type == "table"


def test_detect_time_expressions(analyzer, as_of_date):
    q = "Tổng doanh thu trong tháng này và tuần rồi là bao nhiêu?"
    info = analyzer.analyze(q, as_of_date)
    assert len(info.time_expressions) >= 2
    expr_names = [e[0] for e in info.time_expressions]
    assert any("tháng này" in e for e in expr_names)
    assert any("tuần rồi" in e for e in expr_names)


def test_explicit_quarter_and_year_are_resolved_as_one_period(analyzer):
    info = analyzer.analyze(
        "trong quý 3 năm 2025, doanh thu là bao nhiêu tiền",
        date(2026, 9, 14),
    )

    assert info.aggregation == "sum"
    assert info.time_expressions == [
        ("quý 3 năm 2025", date(2025, 7, 1), date(2025, 9, 30))
    ]
