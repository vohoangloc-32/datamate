import pandas as pd
import pytest
from src.modules.answer_formatter import format_list, resolve_entity_codes_in_answer
from src.modules.accounting_cleaner import (
    _RANK_CUSTOMER_RE,
    _normalize_question,
    patch_code_and_tables_for_rulebook,
)


def test_phase10_format_list_dataframe_and_nested():
    # 1. 1-column DataFrame
    df = pd.DataFrame({"names": ["Sản phẩm A", "Sản phẩm B", "Sản phẩm C"]})
    res = format_list(df)
    assert res == ["Sản phẩm A", "Sản phẩm B", "Sản phẩm C"]

    # 2. Dict with numeric values (e.g. multi-month revenues)
    d = {"2025-04": 287435200, "2025-05": 346554700}
    res = format_list(d)
    assert res == [287435200, 346554700]

    # 3. 1D-like nested list [[A], [B], [C]]
    nested = [["Mai Hoa Nhung"], ["Tạ Thảo Tâm"], ["Ông Hằng Bảo"]]
    res = format_list(nested)
    assert res == ["Mai Hoa Nhung", "Tạ Thảo Tâm", "Ông Hằng Bảo"]

    # 4. Debt 2-tuple with text prefix: ['nợ thêm 80000', 'trả được 40000']
    debt_strs = ["nợ thêm 80.000", "trả được 40.000"]
    res = format_list(debt_strs)
    assert res == [80000, 40000]


def test_phase10_resolve_entity_codes_dynamic_columns():
    # biz_129 schema: Mã khách lẻ/sỉ and KH
    kh_df = pd.DataFrame({
        "Mã khách lẻ/sỉ": ["KH0027", "KH0112"],
        "KH": ["Hoàng Trung Long", "Ngô Thu Hùng"],
    })
    ans = {
        "type": "list",
        "answer": ["KH0027", "KH0112"],
    }
    resolved = resolve_entity_codes_in_answer(
        ans,
        "tháng 6 năm 2026, 5 khách nào chi tiêu nhiều nhất hả",
        {"khach_hang": kh_df},
    )
    assert resolved["answer"] == ["Hoàng Trung Long", "Ngô Thu Hùng"]


def test_phase10_rank_customer_regex():
    q = "5 khach co tong tien mua cao nhat tuan roi la ai"
    norm_q = _normalize_question(q)
    assert bool(_RANK_CUSTOMER_RE.search(norm_q))


def test_phase10_conversion_ratio_fillna():
    code = "df['Số lượng quy đổi'] = df['Số lượng'] * df['Tỉ lệ quy đổi']"
    patched, _ = patch_code_and_tables_for_rulebook(code, "hàng nào bán chạy nhất tháng 4/2025", {})
    assert "df['Tỉ lệ quy đổi'].fillna(1)" in patched


def test_phase10_product_ranking_revenue_redirect():
    code = "df.groupby('Tên hàng')['Thành tiền'].sum().sort_values(ascending=False).head(5)"
    patched, _ = patch_code_and_tables_for_rulebook(code, "tháng 5/2025 top 5 hàng hot nhất là gì", {})
    assert "['Số lượng quy đổi'].sum()" in patched
