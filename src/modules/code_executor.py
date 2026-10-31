"""
code_executor.py — Sandboxed Code Execution for DataMate

Executes LLM-generated Python/pandas code in a restricted environment
with pre-loaded DataFrames, capturing results and handling errors.
"""

import builtins
import io
import logging
import signal
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


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


# Top-level modules that LLM-generated code is allowed to import.
_ALLOWED_IMPORTS = frozenset({
    "pandas", "numpy", "math", "re", "datetime", "calendar", "unicodedata",
    "collections", "itertools", "functools", "statistics", "decimal",
    "fractions", "string", "operator", "typing", "json",
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

    # Load tables as both their original names and sanitized variable names
    for name, df in tables.items():
        # Make a copy to prevent modification of cached data
        df_copy = df.copy()
        safe_globals[name] = df_copy
        # Also create a sanitized variable name (e.g., "sales data" -> "sales_data")
        sanitized = name.replace(" ", "_").replace("-", "_").replace(".", "_")
        if sanitized != name:
            safe_globals[sanitized] = df_copy
        # If there's only one table, also assign it to "df"
        if len(tables) == 1:
            safe_globals["df"] = df_copy

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

        # Build execution environment
        exec_globals = _create_safe_globals(tables, as_of_date)

        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, exec_globals)

            result.stdout = stdout_capture.getvalue()
            result.stderr = stderr_capture.getvalue()

            # Extract results
            result.result = exec_globals.get("result", None)
            result.answer_type = exec_globals.get("answer_type", "abstain")
            result.success = True

            # Validate result
            if result.result is None and result.answer_type not in ("unanswerable", "abstain"):
                result.answer_type = "abstain"
                logger.warning("Code produced None result, setting to abstain")

        except Exception as e:
            result.stdout = stdout_capture.getvalue()
            result.stderr = stderr_capture.getvalue()
            result.error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            result.success = False
            logger.warning("Code execution failed: %s", str(e))

        return result

    def get_table_var_mapping(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Get the mapping from table names to variable names."""
        mapping = {}
        for name in tables:
            sanitized = name.replace(" ", "_").replace("-", "_").replace(".", "_")
            mapping[name] = sanitized
            if len(tables) == 1:
                mapping[name] = "df"
        return mapping

    def get_head_rows(self, tables: Dict[str, pd.DataFrame],
                       n: int = 3) -> Dict[str, str]:
        """Get the first n rows of each table as a string."""
        result = {}
        for name, df in tables.items():
            result[name] = df.head(n).to_string(index=False)
        return result
