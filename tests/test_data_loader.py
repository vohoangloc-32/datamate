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
