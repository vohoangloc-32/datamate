"""
retry_engine.py — Retry & Self-Correction Loop for DataMate

Manages the LLM code generation → execution → retry cycle.
Feeds error messages back to the LLM for self-correction up to N retries.
"""

import ast
import difflib
import logging
import re
import time
import unicodedata
from typing import Callable, Dict, Optional, Tuple

import pandas as pd

from .code_executor import CodeExecutor, ExecutionResult
from .prompt_builder import PromptBuilder, PromptTokenLimitError

logger = logging.getLogger(__name__)


class RetryEngine:
    """
    Manages the code generation → execution → retry loop.

    On failure, feeds the error back to the LLM for self-correction,
    up to max_retries attempts.
    """

    def __init__(
        self,
        code_executor: CodeExecutor,
        prompt_builder: PromptBuilder,
        llm_generate_fn: Callable[[str], str],
        max_retries: int = 3,
    ):
        """
        Parameters
        ----------
        code_executor : CodeExecutor
            The sandboxed code executor.
        prompt_builder : PromptBuilder
            The prompt builder for constructing prompts.
        llm_generate_fn : Callable[[str], str]
            Function that takes a prompt string and returns the LLM response.
        max_retries : int
            Maximum number of retry attempts.
        """
        self.executor = code_executor
        self.prompt_builder = prompt_builder
        self.llm_generate = llm_generate_fn
        self.max_retries = max_retries

    def run(
        self,
        prompt: str,
        tables: Dict[str, pd.DataFrame],
        as_of_date: str,
        expected_type: Optional[str] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        max_prompt_tokens: Optional[int] = None,
        question: str = "",
    ) -> Tuple[ExecutionResult, int]:
        """
        Run the code generation → execution loop with retries.

        Parameters
        ----------
        prompt : str
            The initial prompt for code generation.
        tables : Dict[str, pd.DataFrame]
            Business data tables.
        as_of_date : str
            The as-of date in ISO format.
        expected_type : str, optional
            Expected answer type for validation.
        token_counter : Callable[[str], int], optional
            Counts tokens using the generation model's tokenizer.
        max_prompt_tokens : int, optional
            Maximum input tokens allowed for a retry prompt.
        Returns
        -------
        Tuple[ExecutionResult, int]
            The final execution result and the number of retries used.
        """
        current_prompt = prompt
        best_result: Optional[ExecutionResult] = None
        retries = 0
        last_generation_error: Optional[str] = None
        seen_generation_failures = set()
        seen_execution_failures = set()

        for attempt in range(1 + self.max_retries):
            # Generate code from LLM
            t0 = time.time()
            try:
                llm_response = self.llm_generate(current_prompt)
            except Exception as e:
                logger.error("LLM generation failed on attempt %d: %s", attempt + 1, e)
                last_generation_error = f"{type(e).__name__}: {e}"
                if self._is_context_length_error(e):
                    best_result = ExecutionResult()
                    best_result.answer_type = "abstain"
                    best_result.error = last_generation_error
                    retries = attempt
                    break
                generation_failure = (
                    current_prompt,
                    type(e).__name__,
                    str(e).splitlines()[0],
                )
                if generation_failure in seen_generation_failures:
                    logger.warning(
                        "Stopping after repeated generation failure on attempt %d",
                        attempt + 1,
                    )
                    retries = attempt
                    break
                seen_generation_failures.add(generation_failure)
                retries = attempt
                continue
            gen_time = time.time() - t0
            logger.info("LLM generation attempt %d took %.1fs", attempt + 1, gen_time)

            # Execute the generated code
            t0 = time.time()
            result = self.executor.execute(
                llm_response, tables, as_of_date, extract_code=True, question=question
            )
            exec_time = time.time() - t0
            logger.info("Code execution attempt %d took %.1fs, success=%s",
                        attempt + 1, exec_time, result.success)

            # Check if successful
            if result.success:
                # Validate answer type if expected
                if expected_type and result.answer_type != expected_type:
                    # Type mismatch — retry with hint
                    if attempt < self.max_retries:
                        logger.info(
                            "Type mismatch: expected=%s, got=%s. Retrying.",
                            expected_type, result.answer_type
                        )
                        error_msg = (
                            f"The answer type should be '{expected_type}' but "
                            f"your code produced '{result.answer_type}'. "
                            f"Please fix the code to return the correct type."
                        )
                        try:
                            current_prompt = self.prompt_builder.build_retry_prompt(
                                prompt,
                                result.code,
                                error_msg,
                                available_context=self._available_context(
                                    tables, error_msg
                                ),
                                token_counter=token_counter,
                                max_tokens=max_prompt_tokens,
                            )
                        except PromptTokenLimitError as exc:
                            result.answer_type = "abstain"
                            result.error = str(exc)
                            return (result, attempt)
                        best_result = result
                        retries = attempt + 1
                        continue

                # Validate result is not None (unless unanswerable/abstain)
                if (result.result is None and
                    result.answer_type not in ("unanswerable", "abstain")):
                    if attempt < self.max_retries:
                        logger.info("Result is None but type is %s. Retrying.",
                                    result.answer_type)
                        error_msg = (
                            "Your code set result = None. Please ensure `result` "
                            "is assigned a valid value, or set answer_type = "
                            "'unanswerable' if the data cannot answer the question."
                        )
                        try:
                            current_prompt = self.prompt_builder.build_retry_prompt(
                                prompt,
                                result.code,
                                error_msg,
                                available_context=self._available_context(tables, error_msg),
                                token_counter=token_counter,
                                max_tokens=max_prompt_tokens,
                            )
                        except PromptTokenLimitError as exc:
                            result.answer_type = "abstain"
                            result.error = str(exc)
                            return (result, attempt)
                        best_result = result
                        retries = attempt + 1
                        continue

                # Semantic Validation for Percentage Proportion & Debt Collection
                if attempt < self.max_retries and result.success and question:
                    q_lower = question.lower()

                    # 1. Degenerate Percent Check on category / group share
                    is_proportion_q = any(
                        k in q_lower
                        for k in (
                            "tỉ trọng", "ty trong", "chiếm bao nhiêu %", "chiem bnhiu %",
                            "chiếm bn %", "chiem bao nhieu %", "tỉ lệ dthu", "tỷ lệ dthu", "phần của"
                        )
                    )
                    if is_proportion_q and result.answer_type == "percent":
                        try:
                            val_pct = float(result.result)
                            if val_pct in (100.0, 100, -100.0, -100, 0.0, 0):
                                logger.info(
                                    "Degenerate percent %s for proportion question. Triggering semantic retry (attempt %d).",
                                    val_pct,
                                    attempt + 1,
                                )
                                error_msg = (
                                    f"SemanticWarning: Result produced {val_pct}% for a category proportion question. "
                                    "Rulebook §1 requires that the denominator is the TOTAL revenue of ALL products/categories "
                                    "in the specified period. Do NOT filter the denominator table by category!"
                                )
                                try:
                                    current_prompt = self.prompt_builder.build_retry_prompt(
                                        prompt,
                                        result.code,
                                        error_msg,
                                        available_context=self._available_context(tables, error_msg),
                                        token_counter=token_counter,
                                        max_tokens=max_prompt_tokens,
                                    )
                                    best_result = result
                                    retries = attempt + 1
                                    continue
                                except PromptTokenLimitError:
                                    pass
                        except (ValueError, TypeError):
                            pass

                    # 2. Implausible Debt Collection (Customer debt repayment vs total store receipts)
                    is_debt_payment_q = any(
                        k in q_lower
                        for k in (
                            "thu nợ", "khách trả nợ", "trả bớt nợ", "thu hồi nợ",
                            "thu hoi cong no", "khach tra no", "thu hoi dc bn tien cong no"
                        )
                    )
                    if is_debt_payment_q and result.answer_type == "number":
                        try:
                            val_num = float(result.result)
                            if val_num > 50_000_000:
                                logger.info(
                                    "Implausible debt payment %s (>50M). Triggering semantic retry (attempt %d).",
                                    val_num,
                                    attempt + 1,
                                )
                                error_msg = (
                                    f"Rulebook §4 Guidance: For customer debt repayment ('thu nợ', 'khách trả nợ', 'trả bớt nợ'), "
                                    "only count payments made AFTER the order date (ngày thanh toán > ngày mua) for credit orders. "
                                    "Do NOT sum total receipts/cash flow of the entire store!"
                                )
                                try:
                                    current_prompt = self.prompt_builder.build_retry_prompt(
                                        prompt,
                                        result.code,
                                        error_msg,
                                        available_context=self._available_context(tables, error_msg),
                                        token_counter=token_counter,
                                        max_tokens=max_prompt_tokens,
                                    )
                                    best_result = result
                                    retries = attempt + 1
                                    continue
                                except PromptTokenLimitError:
                                    pass
                        except (ValueError, TypeError):
                            pass

                # Success
                return (result, attempt)

            else:
                # Execution failed — retry with error message
                best_result = result
                retries = attempt

                error_message = result.error or "Unknown execution error"
                failure_signature = (
                    result.code.strip(),
                    error_message.splitlines()[0].strip(),
                )
                if failure_signature in seen_execution_failures:
                    logger.warning(
                        "Stopping after repeated code and error on attempt %d: %s",
                        attempt + 1,
                        failure_signature[1],
                    )
                    break
                seen_execution_failures.add(failure_signature)

                if attempt < self.max_retries:
                    error_msg = error_message
                    # Truncate very long error messages
                    if len(error_msg) > 1000:
                        error_msg = error_msg[:1000] + "\n... (truncated)"

                    try:
                        available_context = self._available_context(
                            tables, error_msg
                        )
                        if "EmptyFilterError" in error_message:
                            suggestions = self._filter_value_suggestions(
                                tables,
                                result.code,
                            )
                            if suggestions:
                                available_context = (
                                    f"{available_context or ''}\n{suggestions}"
                                )
                        current_prompt = self.prompt_builder.build_retry_prompt(
                            prompt,
                            result.code,
                            error_msg,
                            available_context=available_context,
                            token_counter=token_counter,
                            max_tokens=max_prompt_tokens,
                        )
                    except PromptTokenLimitError as exc:
                        result.answer_type = "abstain"
                        result.error = str(exc)
                        break
                    logger.info("Retrying after error: %s", error_msg[:200])

        # All retries exhausted — return best result or abstain
        if best_result is None:
            best_result = ExecutionResult()
            best_result.answer_type = "abstain"
            best_result.error = last_generation_error or "All attempts failed"

        if not best_result.success:
            best_result.answer_type = "abstain"

        return (best_result, retries)

    @staticmethod
    def _is_context_length_error(error: Exception) -> bool:
        message = str(error).casefold()
        return any(marker in message for marker in (
            "maximum context length",
            "max_model_len",
            "input_tokens",
            "context length",
            "exceeds this limit",
        ))

    def _available_context(
        self,
        tables: Dict[str, pd.DataFrame],
        error_message: str,
    ) -> Optional[str]:
        mapping = self.executor.get_table_var_mapping(tables)

        if "NameError" in error_message:
            return self._schema_context(tables, mapping)

        if any(marker in error_message for marker in (
            "KeyError",
            "FileNotFoundError",
            "FileAccessError",
            "EmptyFilterError",
        )):
            context = self._schema_context(
                tables,
                mapping,
                include_dtypes=False,
                include_values=True,
            )
            if "KeyError" in error_message:
                key_hint = self._key_error_hint(tables, error_message)
                if key_hint:
                    context = key_hint + "\n" + context
            if any(marker in error_message for marker in (
                "FileNotFoundError",
                "FileAccessError",
            )):
                context = (
                    "The data is already loaded in these DataFrames. Do not read or "
                    "reload files; use only these tables and columns.\n" + context
                )
            if "EmptyFilterError" in error_message:
                context = (
                    "A numeric result was zero after a DataFrame filter matched "
                    "no rows. Verify every filter against the actual category "
                    "values below before returning zero.\n" + context
                )
            return context[:6000]

        if any(marker in error_message for marker in (
            "Timestamp",
            "datetime.date",
            "not supported between instances of 'str' and 'Timestamp'",
            "DateParseError",
            "ParserError",
            "DateTypeError",
        )):
            context = self._schema_context(tables, mapping, include_dtypes=True)
            context = (
                "Date correction: parse string date columns with "
                "`pd.to_datetime(series, errors='coerce')`, keep values as pandas "
                "timestamps, and compare with `pd.Timestamp(...)`. Do not convert "
                "a whole column with `.dt.date`. Use the actual date columns below.\n"
                + context
            )
            return context[:6000]

        if "SyntaxError" in error_message:
            return (
                "The generated code has a syntax error. Return one complete, valid "
                "Python code block with balanced quotes/brackets and assign both "
                "`result` and `answer_type`."
            )

        if "TypeError" in error_message or "ValueError" in error_message:
            return self._schema_context(tables, mapping, include_dtypes=True)

        return None

    def _filter_value_suggestions(
        self,
        tables: Dict[str, pd.DataFrame],
        code: str,
    ) -> Optional[str]:
        """Suggest observed categories close to string literals used in filters."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None

        table_variables = self.executor.get_table_var_mapping(tables)
        variable_tables = {
            variable: table
            for table, variable in table_variables.items()
        }
        variable_tables.update({
            table: table
            for table in tables
            if table.isidentifier()
        })
        if len(tables) == 1:
            only_table = next(iter(tables))
            variable_tables["df"] = only_table

        def column_reference(node):
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                return variable_tables.get(node.value.id), node.slice.value
            return None, None

        def normalize(value):
            text = unicodedata.normalize("NFKD", str(value).casefold())
            text = "".join(
                character for character in text
                if not unicodedata.combining(character)
            ).replace("đ", "d")
            return re.sub(r"[^a-z0-9]+", " ", text).strip()

        suggestions = []
        seen_suggestions = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            for operator, comparator in zip(node.ops, node.comparators):
                if not isinstance(operator, (ast.Eq, ast.NotEq)):
                    continue
                left_table, left_column = column_reference(node.left)
                if left_table is not None and isinstance(comparator, ast.Constant):
                    table_name, column, literal = (
                        left_table,
                        left_column,
                        comparator.value,
                    )
                else:
                    right_table, right_column = column_reference(comparator)
                    if right_table is None or not isinstance(node.left, ast.Constant):
                        continue
                    table_name, column, literal = (
                        right_table,
                        right_column,
                        node.left.value,
                    )
                if not isinstance(literal, str) or table_name is None:
                    continue

                series = tables[table_name][column]
                if series.nunique(dropna=True) > 50:
                    continue
                values = [str(value) for value in series.dropna().unique()]
                normalized_values = {
                    normalize(value): value for value in values
                }
                normalized_literal = normalize(literal)
                closest = difflib.get_close_matches(
                    normalized_literal,
                    normalized_values,
                    n=1,
                    cutoff=0.72,
                )
                if closest and normalized_values[closest[0]] != literal:
                    suggestion = (
                        f"`{table_name}.{column}` filter {literal!r} is close to "
                        f"observed value {normalized_values[closest[0]]!r}."
                    )
                    if suggestion not in seen_suggestions:
                        seen_suggestions.add(suggestion)
                        suggestions.append(suggestion)

        if not suggestions:
            return None
        return (
            "Closest observed category values for generated filters "
            "(suggestions; verify before using):\n- "
            + "\n- ".join(suggestions[:5])
        )

    @staticmethod
    def _key_error_hint(
        tables: Dict[str, pd.DataFrame],
        error_message: str,
    ) -> Optional[str]:
        """Propose closest existing column names when code fails with KeyError."""
        key_match = re.search(r"KeyError:\s*['\"]([^'\"]+)['\"]", error_message)
        if not key_match:
            return None
        missing_key = key_match.group(1).strip()
        if not missing_key:
            return None

        def normalize(v: str) -> str:
            text = unicodedata.normalize("NFKD", str(v).casefold())
            text = "".join(c for c in text if not unicodedata.combining(c)).replace("đ", "d")
            return re.sub(r"[^a-z0-9]+", "", text)

        norm_missing = normalize(missing_key)
        all_cols = []
        for tname, df in tables.items():
            for col in df.columns:
                if not str(col).startswith("__"):
                    all_cols.append((str(col), tname))

        # Check exact normalized match (case/space/accent-insensitive)
        matches = []
        seen = set()
        for col_name, tname in all_cols:
            if normalize(col_name) == norm_missing and col_name not in seen:
                matches.append(f"`{col_name}` (in `{tname}`)")
                seen.add(col_name)

        # Fallback to difflib
        if not matches:
            unique_cols = list(dict.fromkeys(c for c, _ in all_cols))
            close = difflib.get_close_matches(missing_key, unique_cols, n=3, cutoff=0.5)
            for c in close:
                if c not in seen:
                    for col_name, tname in all_cols:
                        if col_name == c:
                            matches.append(f"`{c}` (in `{tname}`)")
                            seen.add(c)
                            break

        if matches:
            return (
                f"KEYERROR HINT: Column {missing_key!r} does not exist. "
                f"Did you mean: {', '.join(matches[:3])}? Use the exact column name."
            )
        return None

    @staticmethod
    def _schema_context(
        tables: Dict[str, pd.DataFrame],
        mapping: Dict[str, str],
        include_dtypes: bool = False,
        include_values: bool = False,
    ) -> str:
        table_columns = []
        for table, dataframe in tables.items():
            variable = mapping[table]
            column_descriptions = []
            for column in dataframe.columns[:80]:
                series = dataframe[column]
                description = (
                    f"{column} ({series.dtype})"
                    if include_dtypes else str(column)
                )
                if include_values and series.nunique(dropna=True) <= 20:
                    values = [
                        str(value)[:80]
                        for value in series.dropna().unique()[:20]
                    ]
                    if values:
                        description += f" values={values}"
                column_descriptions.append(description)
            columns = ", ".join(column_descriptions)
            if len(dataframe.columns) > 80:
                columns += ", ..."
            table_columns.append(
                f"- `{variable}` (table `{table}`): {columns}"
            )
        return "Available DataFrame variables and actual columns:\n" + "\n".join(
            table_columns
        )
