import pandas as pd
import pytest

from src.modules.value_grounder import ValueGrounder, remove_accents


def test_remove_accents():
    assert remove_accents("Xi măng Hà Tiên") == "Xi mang Ha Tien"
    assert remove_accents("đơn hàng đã hoàn thành") == "don hang da hoan thanh"
    assert remove_accents("Cà phê sữa đá") == "Ca phe sua da"


def test_value_grounder_resolves_exact_and_unaccented_entities():
    tables = {
        "don_hang": pd.DataFrame({
            "ten_sp": [
                "Xi măng Hà Tiên PCB40 bao 50kg",
                "Nước ngọt Sting dâu lon 330ml",
                "Trà xanh C2 hương chanh",
            ],
            "trang_thai": ["hoàn thành", "huỷ", "đang giao"],
            "khach_hang": ["Chị Lan", "Anh Tú (Thợ điện)", "Khách lẻ"],
        })
    }

    grounder = ValueGrounder(tables)

    # 1. Unaccented product lookup: "xi mang ha tien"
    res1 = grounder.resolve_entities("sp bán dc nhiều nhất là xi mang ha tien")
    assert "don_hang" in res1
    assert "ten_sp" in res1["don_hang"]
    assert "Xi măng Hà Tiên PCB40 bao 50kg" in res1["don_hang"]["ten_sp"]

    # 2. Shorthand customer lookup: "anh tu"
    res2 = grounder.resolve_entities("anh tu con no bao nhieu")
    assert "khach_hang" in res2["don_hang"]
    assert "Anh Tú (Thợ điện)" in res2["don_hang"]["khach_hang"]

    # 3. Status lookup: "hoàn thành"
    res3 = grounder.resolve_entities("tổng doanh thu các đơn đã hoàn thành")
    assert "trang_thai" in res3["don_hang"]
    assert "hoàn thành" in res3["don_hang"]["trang_thai"]

    # 4. Formatted context output
    context = grounder.format_grounded_context("xi mang ha tien")
    assert "【Gợi ý giá trị thực tế trong dữ liệu (Entity Grounding)】" in context
    assert "Xi măng Hà Tiên PCB40 bao 50kg" in context
