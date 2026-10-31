"""
retry_engine.py — Retry & Self-Correction Loop for DataMate

Manages the LLM code generation → execution → retry cycle.
Feeds error messages back to the LLM for self-correction up to N retries.
"""

import logging
import time
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd

from .code_executor import CodeExecutor, ExecutionResult
from .prompt_builder import PromptBuilder

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

        Returns
        -------
        Tuple[ExecutionResult, int]
            The final execution result and the number of retries used.
        """
        current_prompt = prompt
        best_result: Optional[ExecutionResult] = None
        retries = 0

        for attempt in range(1 + self.max_retries):
            # Generate code from LLM
            t0 = time.time()
            try:
                llm_response = self.llm_generate(current_prompt)
            except Exception as e:
                logger.error("LLM generation failed on attempt %d: %s", attempt + 1, e)
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
                        current_prompt = self.prompt_builder.build_retry_prompt(
                            prompt, result.code, error_msg
                        )
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
                        current_prompt = self.prompt_builder.build_retry_prompt(
                            prompt, result.code, error_msg
                        )
                        best_result = result
                        retries = attempt + 1
                        continue

                # Success
                return (result, retries)

            else:
                # Execution failed — retry with error message
                best_result = result
                retries = attempt + 1

                if attempt < self.max_retries:
                    error_msg = result.error or "Unknown execution error"
                    # Truncate very long error messages
                    if len(error_msg) > 1000:
                        error_msg = error_msg[:1000] + "\n... (truncated)"

                    current_prompt = self.prompt_builder.build_retry_prompt(
                        prompt, result.code, error_msg
                    )
                    logger.info("Retrying after error: %s", error_msg[:200])

        # All retries exhausted — return best result or abstain
        if best_result is None:
            best_result = ExecutionResult()
            best_result.answer_type = "abstain"
            best_result.error = "All attempts failed"

        if not best_result.success:
            best_result.answer_type = "abstain"

        return (best_result, retries)
