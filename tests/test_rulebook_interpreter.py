"""
Tests for rulebook_interpreter module.
"""

import pytest
from src.modules.rulebook_interpreter import (
    Rulebook,
    BusinessRule,
    _parse_rules_from_text,
    _classify_rule,
    parse_rulebook,
)


@pytest.fixture
def mock_rulebook_text():
    return """
# QUY ĐỊNH CHUNG VỀ KINH DOANH

## 1. Định nghĩa doanh thu
"Doanh thu" = Tổng giá trị các dòng hàng sau khi trừ giảm giá, không bao gồm VAT và phí vận chuyển. Đơn hàng bị hủy không được tính vào doanh thu.

## 2. Tiêu chí xếp hạng
- **Bán chạy**: Được xếp hạng dựa trên tổng số lượng sản phẩm bán ra (quantity), không tính theo doanh thu.
- **Top khách hàng**: Xếp hạng theo tổng giá trị mua hàng tích lũy.

## 3. Thời gian và lịch làm việc
"Tuần" = Bắt đầu từ Thứ Hai và kết thúc vào Chủ Nhật.
"Tháng này" = Từ ngày 1 của tháng cho đến ngày as_of_date hiện tại.

## 4. Giá vốn
- **Giá vốn mặc định**: Tính theo phương pháp bình quân gia quyền theo lô nhập kho (weighted-average).
"""


def test_classify_rule():
    assert _classify_rule("Doanh thu", "Tổng giá trị bán được") == "revenue"
    assert _classify_rule("Bán chạy", "Xếp hạng theo số lượng bán") == "ranking"
    assert _classify_rule("Tuần", "Tính từ thứ hai đến chủ nhật") == "time"
    assert _classify_rule("Giá vốn", "Phương pháp bình quân gia quyền cost") == "cost"


def test_parse_rules_from_text(mock_rulebook_text):
    rules = _parse_rules_from_text(mock_rulebook_text)
    assert len(rules) >= 4

    keywords = [r.keyword.lower() for r in rules]
    assert any("doanh thu" in k for k in keywords)
    assert any("bán chạy" in k for k in keywords)
    assert any("tuần" in k for k in keywords)
    assert any("giá vốn" in k for k in keywords)


def test_rulebook_relevant_rules(mock_rulebook_text):
    rulebook = Rulebook()
    rulebook.rules = _parse_rules_from_text(mock_rulebook_text)

    # Question about best selling products
    q = "Sản phẩm nào bán chạy nhất trong tuần rồi?"
    relevant = rulebook.get_relevant_rules(q)

    rel_keywords = [r.keyword.lower() for r in relevant]
    assert any("bán chạy" in k for k in rel_keywords) or any("tuần" in k for k in rel_keywords)


def test_to_prompt_string(mock_rulebook_text):
    rulebook = Rulebook()
    rulebook.rules = _parse_rules_from_text(mock_rulebook_text)

    prompt_str = rulebook.to_prompt_string()
    assert "Default business rules" in prompt_str
    assert "Doanh thu" in prompt_str or "doanh thu" in prompt_str.lower()


def test_parse_nonexistent_rulebook(tmp_path):
    rb = parse_rulebook(str(tmp_path / "nonexistent.md"))
    assert len(rb.rules) == 0
