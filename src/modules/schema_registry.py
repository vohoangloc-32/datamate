"""
schema_registry.py — Unified Schema Registry for DataMate

Builds a structured metadata store for each business's data tables.
Used downstream for prompt construction, code generation grounding,
and data validation.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
import unicodedata

logger = logging.getLogger(__name__)


class ColumnInfo:
    """Metadata for a single column."""

    __slots__ = ("name", "dtype", "non_null_count", "total_count",
                 "unique_count", "sample_values", "min_val", "max_val",
             "is_numeric", "is_datetime", "is_categorical",
             "category_values")

    def __init__(self, name: str, series: pd.Series):
        self.name = name
        self.dtype = str(series.dtype)
        self.total_count = len(series)
        self.non_null_count = int(series.notna().sum())
        self.unique_count = int(series.nunique())

        non_null = series.dropna()
        self.sample_values = non_null.head(5).tolist()

        self.is_numeric = pd.api.types.is_numeric_dtype(series)
        self.is_datetime = pd.api.types.is_datetime64_any_dtype(series)
        self.is_categorical = (
            not self.is_numeric
            and not self.is_datetime
            and self.unique_count <= 50
            and self.non_null_count > 0
        )
        self.category_values = (
            non_null.unique().tolist()
            if self.is_categorical and self.unique_count <= 20
            else []
        )

        if self.is_numeric and self.non_null_count > 0:
            self.min_val = float(non_null.min())
            self.max_val = float(non_null.max())
        elif self.is_datetime and self.non_null_count > 0:
            self.min_val = str(non_null.min())
            self.max_val = str(non_null.max())
        else:
            self.min_val = None
            self.max_val = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "non_null": self.non_null_count,
            "total": self.total_count,
            "unique": self.unique_count,
            "samples": self.sample_values[:3],
            "min": self.min_val,
            "max": self.max_val,
            "is_numeric": self.is_numeric,
            "is_datetime": self.is_datetime,
            "is_categorical": self.is_categorical,
            "categories": self.category_values,
        }


class TableSchema:
    """Schema for a single table/sheet."""

    def __init__(self, name: str, df: pd.DataFrame):
        self.name = name
        self.row_count = len(df)
        self.columns: List[ColumnInfo] = [
            ColumnInfo(col, df[col]) for col in df.columns
        ]
        self.column_names = [c.name for c in self.columns]

    def get_column(self, name: str) -> Optional[ColumnInfo]:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    def to_prompt_string(self, include_samples: bool = True, max_sample_values: int = 5) -> str:
        """Generate a concise schema description for LLM prompts."""
        lines = [f"Table: {self.name} ({self.row_count} rows)"]
        lines.append("Columns:")
        for col in self.columns:
            parts = [f"  - {col.name} ({col.dtype})"]
            if col.is_categorical and col.unique_count <= 20:
                vals = col.category_values
                parts.append(f" [categories: {vals}]")
            elif include_samples and col.sample_values:
                vals = col.sample_values[:3]
                parts.append(f" [e.g. {vals}]")
            if col.min_val is not None:
                parts.append(f" [range: {col.min_val} – {col.max_val}]")
            lines.append("".join(parts))
        return "\n".join(lines)


class SchemaRegistry:
    """Registry holding schemas for all tables in a business's data."""

    def __init__(self):
        self.tables: Dict[str, TableSchema] = {}

    def register(self, name: str, df: pd.DataFrame):
        """Register a table into the schema registry."""
        schema = TableSchema(name, df)
        self.tables[name] = schema
        logger.info("Registered table '%s': %d rows, %d columns",
                     name, schema.row_count, len(schema.columns))

    def build_from_dict(self, tables: Dict[str, pd.DataFrame]) -> "SchemaRegistry":
        """Build the registry from a dict of {name: DataFrame}."""
        for name, df in tables.items():
            self.register(name, df)
        return self

    def get_table(self, name: str) -> Optional[TableSchema]:
        return self.tables.get(name)

    def table_names(self) -> List[str]:
        return list(self.tables.keys())

    def get_table_names(self) -> List[str]:
        """Alias for table_names."""
        return self.table_names()

    def to_prompt_string(self, include_samples: bool = True) -> str:
        """Generate a full schema description for LLM prompts."""
        parts = []
        for schema in self.tables.values():
            parts.append(schema.to_prompt_string(include_samples=include_samples))
        return "\n\n".join(parts)

    def to_compact_string(self) -> str:
        """Generate a compact schema (just table names and column names+types)."""
        parts = []
        for schema in self.tables.values():
            cols = ", ".join(f"{c.name}({c.dtype})" for c in schema.columns)
            parts.append(f"{schema.name} [{schema.row_count} rows]: {cols}")
        return "\n".join(parts)

    def find_date_columns(self) -> Dict[str, List[str]]:
        """Find all datetime columns across all tables."""
        result = {}
        for name, schema in self.tables.items():
            date_cols = [c.name for c in schema.columns if c.is_datetime]
            if date_cols:
                result[name] = date_cols
        return result

    def find_numeric_columns(self) -> Dict[str, List[str]]:
        """Find all numeric columns across all tables."""
        result = {}
        for name, schema in self.tables.items():
            num_cols = [c.name for c in schema.columns if c.is_numeric]
            if num_cols:
                result[name] = num_cols
        return result
