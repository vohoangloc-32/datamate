"""
Tests for schema_registry module.
"""

import pytest
import pandas as pd
from src.modules.schema_registry import (
    ColumnInfo,
    TableSchema,
    SchemaRegistry,
)


@pytest.fixture
def sample_data():
    df_products = pd.DataFrame({
        "product_id": [1, 2, 3],
        "category": ["Beverage", "Food", "Beverage"],
        "price": [25000.0, 45000.0, 30000.0],
    })
    df_orders = pd.DataFrame({
        "order_id": [101, 102],
        "amount": [50000.0, 75000.0],
    })
    return {"products": df_products, "orders": df_orders}


def test_column_info():
    s = pd.Series([10.0, 20.0, 50.0])
    col = ColumnInfo("price", s)
    assert col.name == "price"
    assert col.is_numeric is True
    assert col.min_val == 10.0
    assert col.max_val == 50.0
    assert col.unique_count == 3


def test_table_schema(sample_data):
    schema = TableSchema("products", sample_data["products"])
    assert schema.name == "products"
    assert schema.row_count == 3
    assert "price" in schema.column_names

    price_col = schema.get_column("price")
    assert price_col is not None
    assert price_col.is_numeric is True

    prompt_str = schema.to_prompt_string()
    assert "Table: products" in prompt_str
    assert "price" in prompt_str


def test_schema_registry(sample_data):
    registry = SchemaRegistry().build_from_dict(sample_data)
    assert len(registry.tables) == 2
    assert "products" in registry.get_table_names()
    assert "orders" in registry.get_table_names()

    prompt_str = registry.to_prompt_string()
    assert "products" in prompt_str
    assert "orders" in prompt_str


def test_schema_includes_all_low_cardinality_values():
    schema = TableSchema(
        "orders",
        pd.DataFrame({"status": ["Completed", "Cancelled", "Returned"]}),
    )

    prompt = schema.to_prompt_string()

    assert "Completed" in prompt
    assert "Cancelled" in prompt
    assert "Returned" in prompt
