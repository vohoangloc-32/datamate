"""
Tests for data_loader module.
"""

import pytest
import sqlite3
import pandas as pd
from pathlib import Path
from src.modules.data_loader import (
    load_business_data,
    clear_cache,
    _normalize_col_name,
    _coerce_types,
    get_table_info,
)


def test_normalize_col_name():
    assert _normalize_col_name("  Tên   sản phẩm  ") == "Tên sản phẩm"
    assert _normalize_col_name("Order_ID\n") == "Order_ID"


def test_coerce_types():
    df = pd.DataFrame({
        "num_col": ["1", "2", "3", "invalid"],
        "date_col": ["2024-01-01", "2024-01-02", "2024-01-03", "text"],
        "str_col": ["apple", "banana", "orange", "grape"],
    })
    coerced = _coerce_types(df)
    assert pd.api.types.is_numeric_dtype(coerced["num_col"])
    assert pd.api.types.is_datetime64_any_dtype(coerced["date_col"])
    assert pd.api.types.is_string_dtype(coerced["str_col"]) or coerced["str_col"].dtype == object


def test_coerce_excel_serials_and_vietnamese_date_strings():
    df = pd.DataFrame({
        "ngay_ban": [45717.0, 45718.0, None],
        "order_date": ["1 thg 6 2025", "2 thg 6 2025", None],
        "amount": [45658, 45689, 45717],
    })

    coerced = _coerce_types(df)

    assert pd.api.types.is_datetime64_any_dtype(coerced["ngay_ban"])
    assert coerced["ngay_ban"].iloc[0] == pd.Timestamp("2025-03-01")
    assert pd.isna(coerced["ngay_ban"].iloc[2])
    assert pd.api.types.is_datetime64_any_dtype(coerced["order_date"])
    assert coerced["order_date"].iloc[0] == pd.Timestamp("2025-06-01")
    assert pd.api.types.is_numeric_dtype(coerced["amount"])


def test_coerce_mixed_date_column_values_to_timestamps():
    df = pd.DataFrame({
        "Ngày giao dịch": ["2025-06-01", "không rõ", None],
    })

    coerced = _coerce_types(df)

    assert pd.api.types.is_datetime64_any_dtype(coerced["Ngày giao dịch"])
    assert coerced["Ngày giao dịch"].iloc[0] == pd.Timestamp("2025-06-01")
    assert pd.isna(coerced["Ngày giao dịch"].iloc[1])
    assert pd.isna(coerced["Ngày giao dịch"].iloc[2])


def test_date_column_does_not_interpret_year_numbers_as_excel_serials():
    df = pd.DataFrame({"date": [2024, 2025, 2026]})

    coerced = _coerce_types(df)

    assert pd.api.types.is_numeric_dtype(coerced["date"])
    assert coerced["date"].tolist() == [2024, 2025, 2026]


def test_load_csv_and_sqlite(tmp_path):
    biz_dir = tmp_path / "biz_001"
    biz_dir.mkdir()

    # 1. Create a sample CSV
    csv_file = biz_dir / "customers.csv"
    csv_file.write_text("cust_id,cust_name,points\n1,Nguyen Van A,150\n2,Tran Thi B,200\n", encoding="utf-8")

    # 2. Create a sample SQLite db
    db_file = biz_dir / "database.sqlite"
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE orders (order_id INTEGER, total REAL)")
    conn.execute("INSERT INTO orders VALUES (101, 550000.0), (102, 120000.0)")
    conn.commit()
    conn.close()

    # Clear cache before loading
    clear_cache()
    tables = load_business_data(str(biz_dir), use_cache=True)

    assert "customers" in tables
    assert "orders" in tables
    assert len(tables["customers"]) == 2
    assert len(tables["orders"]) == 2
    assert tables["customers"].iloc[0]["cust_name"] == "Nguyen Van A"

    # Test cache hit
    tables_cached = load_business_data(str(biz_dir), use_cache=True)
    assert tables_cached is tables

    # Test get_table_info formatting
    info_str = get_table_info(tables)
    assert "Table: customers" in info_str
    assert "Table: orders" in info_str


def test_load_xlsx_with_report_preamble(tmp_path):
    biz_dir = tmp_path / "biz_report"
    biz_dir.mkdir()
    xlsx_file = biz_dir / "sales.xlsx"
    pd.DataFrame([
        ["Sales report"],
        ["Reporting period", "All"],
        [],
        ["Date", "Customer", "Amount"],
        ["2025-01-01", "Customer A", 1250],
        ["2025-01-02", "Customer B", 2500],
    ]).to_excel(xlsx_file, header=False, index=False)

    tables = load_business_data(str(biz_dir), use_cache=False)
    sales = tables["sales"]

    assert sales.columns.tolist() == ["Date", "Customer", "Amount"]
    assert len(sales) == 2
    assert sales.iloc[0]["Customer"] == "Customer A"


def test_load_xlsx_fills_merged_order_metadata_within_each_order(tmp_path):
    biz_dir = tmp_path / "biz_merged_orders"
    biz_dir.mkdir()
    pd.DataFrame([
        ["ORD-1", "2025-01-01", "Completed", "Customer A", "Item 1", 10],
        [None, None, None, None, "Item 2", 20],
        ["ORD-2", "2025-01-02", "Cancelled", None, "Item 3", 30],
        [None, None, None, None, "Item 4", 40],
    ], columns=[
        "Mã đơn hàng",
        "Ngày đặt hàng",
        "Trạng thái đơn hàng",
        "Tên khách hàng",
        "Tên sản phẩm",
        "Tổng tiền",
    ]).to_excel(biz_dir / "orders.xlsx", index=False)

    orders = load_business_data(str(biz_dir), use_cache=False)["orders"]

    assert orders["Mã đơn hàng"].tolist() == ["ORD-1", "ORD-1", "ORD-2", "ORD-2"]
    assert orders["Ngày đặt hàng"].tolist() == [
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-01"),
        pd.Timestamp("2025-01-02"),
        pd.Timestamp("2025-01-02"),
    ]
    assert orders["Trạng thái đơn hàng"].tolist() == [
        "Completed",
        "Completed",
        "Cancelled",
        "Cancelled",
    ]
    assert pd.isna(orders["Tên khách hàng"].iloc[2])
    assert pd.isna(orders["Tên khách hàng"].iloc[3])


def test_load_xlsx_combines_compatible_monthly_sheets(tmp_path):
    biz_dir = tmp_path / "biz_monthly"
    biz_dir.mkdir()
    xlsx_file = biz_dir / "sales.xlsx"
    with pd.ExcelWriter(xlsx_file) as writer:
        pd.DataFrame({
            "order_id": [1, 2],
            "amount": [100, 200],
        }).to_excel(writer, sheet_name="T01_2025", index=False)
        pd.DataFrame({
            "order_id": [3],
            "amount": [300],
        }).to_excel(writer, sheet_name="02.2025", index=False)

    tables = load_business_data(str(biz_dir), use_cache=False)
    combined = tables["sales__monthly_combined"]

    assert len(combined) == 3
    assert combined["order_id"].tolist() == [1, 2, 3]
    assert combined["__source_month"].tolist() == [
        "2025-01",
        "2025-01",
        "2025-02",
    ]
    assert combined["__source_sheet"].tolist() == [
        "T01_2025",
        "T01_2025",
        "02.2025",
    ]
