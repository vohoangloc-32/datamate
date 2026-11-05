"""
code_executor.py — Sandboxed Code Execution for DataMate

Executes LLM-generated Python/pandas code in a restricted environment
with pre-loaded DataFrames, capturing results and handling errors.
"""

import builtins
import ast
import io
import keyword
import logging
from numbers import Number
import re
import signal
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .accounting_cleaner import (
    patch_code_and_tables_for_rulebook,
    question_is_average_order_value,
    round_half_up_vnd,
)

logger = logging.getLogger(__name__)

_RESERVED_EXEC_NAMES = frozenset({
    "pd", "np", "math", "datetime", "dt", "date", "timedelta", "re",
    "as_of_date", "as_of_date_obj", "result", "answer_type", "__builtins__",
})

_FILE_READ_CALLS = frozenset({
    "open",
    "read_csv",
    "read_excel",
    "read_fwf",
    "read_json",
    "read_html",
    "read_xml",
    "read_pickle",
    "read_parquet",
    "read_feather",
    "read_hdf",
    "read_stata",
    "read_sas",
    "read_sql",
    "read_sql_query",
    "read_sql_table",
    "read_table",
    "read_clipboard",
    "ExcelFile",
    "HDFStore",
})


def _table_variable_mapping(tables: Dict[str, pd.DataFrame]) -> Dict[str, str]:
    """Map table names to unique, valid Python identifiers."""
    mapping = {}
    used_names = set(_RESERVED_EXEC_NAMES)
    for name in tables:
        candidate = re.sub(r"\W", "_", name, flags=re.UNICODE)
        if not candidate or candidate[0].isdigit():
            candidate = f"table_{candidate}"
        if not candidate.isidentifier() or keyword.iskeyword(candidate):
            candidate = f"{candidate}_table"
        base_name = candidate
        suffix = 2
        while candidate in used_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        mapping[name] = candidate
        used_names.add(candidate)
    return mapping


class ExecutionResult:
    """Result of a code execution."""

    def __init__(self):
        self.success: bool = False
        self.result: Any = None
        self.answer_type: str = "abstain"
        self.stdout: str = ""
        self.stderr: str = ""
        self.error: Optional[str] = None
        self.code: str = ""
        self.empty_dataframes: list[str] = []

    def __repr__(self):
        return (f"ExecutionResult(success={self.success}, "
                f"type={self.answer_type}, result={self.result!r})")


def _extract_code_from_response(response: str) -> str:
    """
    Extract Python code from the LLM response.
    Handles code blocks (```python...```) and raw code.
    """
    # Try to find a python code block
    import re
    patterns = [
        r"```python\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    for pattern in patterns:
        m = re.search(pattern, response, re.DOTALL)
        if m:
            return m.group(1).strip()

    # If no code block, try to find lines that look like code
    lines = response.strip().split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        if (stripped.startswith("import ") or stripped.startswith("from ") or
            stripped.startswith("result") or stripped.startswith("answer_type") or
            stripped.startswith("df") or stripped.startswith("#") or
            "=" in stripped or stripped.startswith("for ") or
            stripped.startswith("if ") or stripped.startswith("    ") or
            in_code):
            code_lines.append(line)
            in_code = True
        elif in_code and not stripped:
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

    # Fallback: return the entire response as code
    return response.strip()


def _file_read_violation(code: str) -> Optional[str]:
    """Return a message when generated code calls a common file-reading API."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in _FILE_READ_CALLS:
            return f"File access is not allowed: `{node.func.id}()`"
        if isinstance(node.func, ast.Attribute) and node.func.attr in _FILE_READ_CALLS:
            return f"File access is not allowed: `.{node.func.attr}()`"
    return None


# Top-level modules that LLM-generated code is allowed to import.
_ALLOWED_IMPORTS = frozenset({
    "pandas", "numpy", "math", "re", "datetime", "calendar", "unicodedata",
    "collections", "itertools", "functools", "statistics", "decimal",
    "fractions", "string", "operator", "typing", "json", "time",
})


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Restricted replacement for ``__import__`` (whitelisted modules only)."""
    root = name.split(".")[0]
    if level != 0 or root not in _ALLOWED_IMPORTS:
        raise ImportError(f"Import of '{name}' is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _create_safe_globals(tables: Dict[str, pd.DataFrame],
                          as_of_date: str) -> Dict[str, Any]:
    """Create the execution globals with pre-loaded DataFrames and safe builtins."""
    import datetime as dt
    import math
    import numpy as np

    safe_globals = {
        "__builtins__": {
            "__import__": _safe_import,
            "abs": abs,
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "format": format,
            "frozenset": frozenset,
            "hasattr": hasattr,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "print": print,
            "range": range,
            "repr": repr,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
            "type": type,
            "zip": zip,
            "None": None,
            "True": True,
            "False": False,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "ZeroDivisionError": ZeroDivisionError,
            "Exception": Exception,
            "AttributeError": AttributeError,
            "ImportError": ImportError,
            "NameError": NameError,
            "getattr": getattr,
            "pow": pow,
            "divmod": divmod,
            "next": next,
            "iter": iter,
            "callable": callable,
        },
        # Standard libraries
        "pd": pd,
        "np": np,
        "math": math,
        "datetime": dt,
        "dt": dt,
        "date": dt.date,
        "timedelta": dt.timedelta,
        "re": __import__("re"),
        # As-of date
        "as_of_date": as_of_date,
        "as_of_date_obj": dt.date.fromisoformat(as_of_date),
        # Result placeholders
        "result": None,
        "answer_type": "abstain",
    }

    # Load tables under their original valid names and unique sanitized names.
    variable_names = _table_variable_mapping(tables)
    for name, df in tables.items():
        # Make a copy to prevent modification of cached data
        df_copy = df.copy()
        variable_name = variable_names[name]
        safe_globals[variable_name] = df_copy
        if (
            variable_name == name
            and name.isidentifier()
            and not keyword.iskeyword(name)
            and name not in _RESERVED_EXEC_NAMES
        ):
            safe_globals[name] = df_copy

    if len(tables) == 1:
        only_table = next(iter(tables))
        safe_globals["df"] = safe_globals[variable_names[only_table]]

    return safe_globals


class CodeExecutor:
    """Execute LLM-generated code in a sandboxed environment."""

    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds

    def execute(
        self,
        code_or_response: str,
        tables: Dict[str, pd.DataFrame],
        as_of_date: str,
        extract_code: bool = True,
        question: str = "",
    ) -> ExecutionResult:
        """
        Execute code with pre-loaded DataFrames.

        Parameters
        ----------
        code_or_response : str
            Either raw Python code or an LLM response containing code.
        tables : Dict[str, pd.DataFrame]
            Business data tables to make available in the execution env.
        as_of_date : str
            The as-of date in ISO format.
        extract_code : bool
            If True, attempt to extract code from the response.

        Returns
        -------
        ExecutionResult
            The execution result with answer, type, stdout/stderr.
        """
        result = ExecutionResult()

        # Extract code from LLM response
        if extract_code:
            code = _extract_code_from_response(code_or_response)
        else:
            code = code_or_response
        result.code = code

        if not code.strip():
            result.error = "No code to execute"
            return result

        file_read_violation = _file_read_violation(code)
        if file_read_violation:
            result.error = f"FileAccessError: {file_read_violation}"
            return result

        exec_code, exec_tables = (
            patch_code_and_tables_for_rulebook(code, question, tables)
            if question
            else (code, tables)
        )

        # Build execution environment
        exec_globals = _create_safe_globals(exec_tables, as_of_date)
        original_dataframe_ids = {
            id(value)
            for value in exec_globals.values()
            if isinstance(value, pd.DataFrame)
        }

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(exec_code, exec_globals)
            except Exception:
                if exec_code != code or exec_tables is not tables:
                    stdout_capture = io.StringIO()
                    stderr_capture = io.StringIO()
                    exec_globals = _create_safe_globals(tables, as_of_date)
                    original_dataframe_ids = {
                        id(value)
                        for value in exec_globals.values()
                        if isinstance(value, pd.DataFrame)
                    }
                    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                        exec(code, exec_globals)
                else:
                    raise

            result.stdout = stdout_capture.getvalue()
            result.stderr = stderr_capture.getvalue()

            # Extract results
            raw_res = exec_globals.get("result", None)
            if question and question_is_average_order_value(question):
                raw_res = round_half_up_vnd(raw_res)
            result.result = raw_res
            result.answer_type = exec_globals.get("answer_type", "abstain")
            result.success = True
            result.empty_dataframes = [
                name
                for name, value in exec_globals.items()
                if isinstance(value, pd.DataFrame)
                and id(value) not in original_dataframe_ids
                and value.empty
            ]

            # Validate result
            if result.result is None and result.answer_type not in ("unanswerable", "abstain"):
                result.answer_type = "abstain"
                logger.warning("Code produced None result, setting to abstain")
            elif (
                result.answer_type in ("number", "percent")
                and isinstance(result.result, Number)
                and not isinstance(result.result, bool)
                and result.result == 0
                and result.empty_dataframes
            ):
                result.error = (
                    "EmptyFilterError: numeric zero was produced while derived "
                    f"DataFrames were empty: {result.empty_dataframes}"
                )
                result.success = False

        except Exception as e:
            result.stdout = stdout_capture.getvalue()
            result.stderr = stderr_capture.getvalue()
            result.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result.success = False
            logger.warning("Code execution failed: %s", str(e))

        return result

    def get_table_var_mapping(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Get the mapping from table names to variable names."""
        mapping = _table_variable_mapping(tables)
        if len(mapping) == 1:
            only_table = next(iter(mapping))
            mapping[only_table] = "df"
        return mapping

    def get_head_rows(self, tables: Dict[str, pd.DataFrame],
                       n: int = 3) -> Dict[str, str]:
        """Get the first n rows of each table as a string."""
        result = {}
        for name, df in tables.items():
            result[name] = df.head(n).to_string(index=False)
        return result
