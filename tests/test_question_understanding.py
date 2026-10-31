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
