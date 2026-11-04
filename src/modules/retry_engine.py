"""
retry_engine.py — Retry & Self-Correction Loop for DataMate

Manages the LLM code generation → execution → retry cycle.
Feeds error messages back to the LLM for self-correction up to N retries.
"""

import logging
import time
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
                llm_response, tables, as_of_date, extract_code=True
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
        )):
            context = self._schema_context(tables, mapping, include_dtypes=False)
            if any(marker in error_message for marker in (
                "FileNotFoundError",
                "FileAccessError",
            )):
                context = (
                    "The data is already loaded in these DataFrames. Do not read or "
                    "reload files; use only these tables and columns.\n" + context
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
                "`result` and `answer_type`. Do not use dotted or hyphenated table "
                "names as Python variables; use the exact valid identifiers listed "
                "below. Do not return Markdown fences inside the code.\n"
                + self._schema_context(tables, mapping)
            )

        if "TypeError" in error_message or "ValueError" in error_message:
            return self._schema_context(tables, mapping, include_dtypes=True)

        return None

    @staticmethod
    def _schema_context(
        tables: Dict[str, pd.DataFrame],
        mapping: Dict[str, str],
        include_dtypes: bool = False,
    ) -> str:
        table_columns = []
        for table, dataframe in tables.items():
            variable = mapping[table]
            columns = ", ".join(
                f"{column} ({dataframe[column].dtype})"
                if include_dtypes else str(column)
                for column in dataframe.columns[:80]
            )
            if len(dataframe.columns) > 80:
                columns += ", ..."
            table_columns.append(
                f"- `{variable}` (table `{table}`): {columns}"
            )
        return "Available DataFrame variables and actual columns:\n" + "\n".join(
            table_columns
        )
