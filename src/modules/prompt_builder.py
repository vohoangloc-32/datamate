"""
prompt_builder.py — LLM Prompt Constructor for DataMate

Builds structured prompts for the code-generation LLM, incorporating:
- Data schema
- Business README context
- Rulebook rules
- Calendar info
- Question + as_of_date
- Few-shot examples
"""

import ast
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .question_understanding import QuestionInfo
from .readme_parser import BusinessContext
from .rulebook_interpreter import Rulebook
from .schema_registry import SchemaRegistry

logger = logging.getLogger(__name__)


class PromptTokenLimitError(ValueError):
    """Raised when a prompt cannot fit the configured model input budget."""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Vietnamese data analyst. Your task is to answer a question about a business's data by writing Python pandas code.

IMPORTANT RULES:
1. The data is already loaded as pandas DataFrames in variables. Use only the exact variable names and column names listed in AVAILABLE DATA. Example names are illustrative and are not available unless listed there.
2. Do not read or reload data from files. The provided DataFrames are the only data sources.
3. Write Python code that computes the answer and assigns it to a variable called `result`.
4. `result` must be a single value matching the expected answer type.
5. If a required table or field is absent from AVAILABLE DATA, do not guess its name or substitute another field. Use `unanswerable` only after checking the available data.
6. Apply the business rules provided below (e.g., how to compute revenue, what counts as a "week", etc.).
7. Handle edge cases: empty DataFrames, missing values, division by zero.
8. For date filtering, use the as_of_date to resolve relative time expressions.
   Keep pandas datetime columns as timestamps; compare with pd.Timestamp values.
   Do not convert a whole datetime column to Python date objects with .dt.date.
9. Do NOT use print() — just assign to `result`.
10. For string answers, return the exact name as it appears in the data.
11. For numeric answers, return an integer if exact, or float rounded to 2 decimals.
12. For percentage answers, return the numeric value (e.g., 12.45 not 0.1245).
13. For date answers, return ISO format string "YYYY-MM-DD" or "YYYY-MM".
14. For boolean answers, return True or False.
15. For list answers, return a Python list in the order requested.
16. For table answers, return a dict with "columns" (list of strings) and "rows" (list of lists).
17. Do NOT write any import statements. `pd`, `np`, `math`, `re`, `datetime`, `dt`, `date` and `timedelta` are already available.
18. Define every variable before using it; the code must be complete and self-contained.
19. Preserve every explicit question entity/filter; never answer an entity-specific count with a generic count.
20. For date ranges spanning same-schema monthly tables, combine all matching tables. Order counts use distinct order IDs, not rows, unless the question asks for order lines/items.
21. Sold-order counts and revenue/customer spending exclude cancelled, returned, or refunded transactions unless asked otherwise. Use actual status values in the data; follow documented customer aliases and do not rank unidentified guests as named customers.
22. For a status rate, count matching-status records in the numerator and all applicable records in the denominator.
23. Use half-open date ranges (`>= start` and `< next_period_start`) for months/quarters.
24. Before writing code, resolve the question's metric, entity, date column, date range, filters, aggregation, grouping, ranking, and output type from the structured plan and rulebook. Every explicit condition must appear in the code.
25. Treat the structured query plan as a contract, not a suggestion. If it is validated, do not change its metric, aggregation, date range, source tables, or answer type.
26. Distinguish an empty result from unavailable data. An empty result can be a valid zero, empty list, or empty table; use `unanswerable` only when the required information or rule is genuinely absent.
27. Distinguish system failures from unavailable data. Syntax errors, missing generated variables, timeouts, memory errors, and execution failures must be fixed or left for retry; never return `unanswerable` for them.
28. For joins, identify the real join keys and check expected uniqueness before aggregating. Do not join detail tables in a way that duplicates monetary or quantity values.
29. When the question asks for a count of orders, use the actual order-ID column and `nunique()` after applying the order-level filters. Do not use row count unless the question asks for rows or lines.
30. Do not infer business meaning from a similarly named column. Prefer the rulebook/README definition and the exact schema; if a required binding remains unresolved, do not invent it.

OUTPUT FORMAT:
Your response must be a Python code block that:
- Uses the available DataFrame variables
- Computes the answer
- Assigns `result` = the answer value
- Assigns `answer_type` = one of: "number", "percent", "string", "date", "boolean", "list", "table", "unanswerable"
- Return only the code block; do not include an explanation, plan prose, or Markdown outside the code.
"""

_PROMPT_GLOBAL_NAMES = frozenset({
    "pd", "np", "math", "re", "datetime", "dt", "date", "timedelta",
    "as_of_date", "as_of_date_obj", "result", "answer_type", "abs", "all",
    "any", "bool", "dict", "enumerate", "filter", "float", "format",
    "frozenset", "hasattr", "int", "isinstance", "len", "list", "map",
    "max", "min", "print", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "type", "zip", "None",
    "True", "False", "ValueError", "TypeError", "KeyError", "IndexError",
    "ZeroDivisionError", "Exception", "AttributeError", "ImportError",
    "NameError", "getattr", "pow", "divmod", "next", "iter", "callable",
})


def _example_matches_schema(
    example: Dict,
    schema: SchemaRegistry,
    table_var_names: Optional[Dict[str, str]],
) -> bool:
    """Avoid showing examples that teach unavailable variables or columns."""
    try:
        tree = ast.parse(example.get("code", ""))
    except SyntaxError:
        return False

    assigned_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    loaded_names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    referenced_globals = loaded_names - assigned_names - _PROMPT_GLOBAL_NAMES
    available_variables = set(table_var_names.values()) if table_var_names else set()
    available_variables.update(schema.table_names())
    if len(schema.tables) == 1:
        available_variables.add("df")
    if not referenced_globals.issubset(available_variables):
        return False

    available_columns = {
        column_name
        for table_schema in schema.tables.values()
        for column_name in table_schema.column_names
    }
    referenced_columns = {
        node.slice.value
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        )
    }
    return referenced_columns.issubset(available_columns)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """Builds structured prompts for the LLM."""

    def __init__(self, few_shot_examples: Optional[List[Dict]] = None):
        if few_shot_examples is None:
            default_path = Path(__file__).resolve().parents[2] / "prompts" / "templates" / "few_shot_examples.json"
            if default_path.exists():
                try:
                    with open(default_path, "r", encoding="utf-8") as f:
                        self.few_shot_examples = json.load(f)
                except Exception as e:
                    logger.warning("Could not load default few-shot examples: %s", e)
                    self.few_shot_examples = []
            else:
                self.few_shot_examples = []
        else:
            self.few_shot_examples = few_shot_examples

    def build(
        self,
        question: str,
        as_of_date: str,
        schema: SchemaRegistry,
        readme_context: Optional[BusinessContext] = None,
        rulebook: Optional[Rulebook] = None,
        question_info: Optional[QuestionInfo] = None,
        table_var_names: Optional[Dict[str, str]] = None,
        head_rows: Optional[Dict[str, str]] = None,
        completeness_context: Optional[Dict[str, Any]] = None,
        query_plan: Optional[Dict[str, Any]] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Build a complete prompt for the code-generation LLM.

        Parameters
        ----------
        question : str
            The Vietnamese question.
        as_of_date : str
            The as-of date in ISO format.
        schema : SchemaRegistry
            Schema information for the business's data.
        readme_context : BusinessContext, optional
            Parsed README context.
        rulebook : Rulebook, optional
            Parsed business rulebook.
        question_info : QuestionInfo, optional
            Pre-analyzed question info.
        table_var_names : Dict[str, str], optional
            Mapping from table name to variable name in the execution env.
        head_rows : Dict[str, str], optional
            First few rows of each table as string (for context).
        token_counter : Callable[[str], int], optional
            Counts tokens using the tokenizer used by the generation model.
        max_tokens : int, optional
            Maximum input tokens allowed for the returned prompt.

        Returns
        -------
        str
            The complete prompt string.
        """
        if (token_counter is None) != (max_tokens is None):
            raise ValueError("token_counter and max_tokens must be provided together.")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be greater than zero.")

        selected_examples = []
        if question_info and question_info.expected_type:
            selected_examples.extend(
                example for example in self.few_shot_examples
                if example.get("expected_type") == question_info.expected_type
            )
        selected_examples = [
            example for example in selected_examples
            if _example_matches_schema(example, schema, table_var_names)
        ][:2]

        def render(
            include_rows: bool,
            include_examples: bool,
            include_context: bool,
            schema_style: str,
            rulebook_limit: int,
            context_limit: int,
        ) -> str:
            sections = [SYSTEM_PROMPT]

            if include_context and readme_context and readme_context.raw_text:
                context = readme_context.to_prompt_string(max_length=context_limit)
                if context:
                    sections.extend(["## BUSINESS CONTEXT", context])

            if rulebook and rulebook.rules and rulebook_limit:
                sections.append("\n## BUSINESS RULES")
                sections.append(rulebook.to_prompt_string(
                    relevant_only=question if question_info else None,
                    max_length=rulebook_limit,
                ))

            sections.append("\n## AVAILABLE DATA")
            if table_var_names:
                sections.append("Variables available in your code (use these exact names):")
                for table_name, var_name in table_var_names.items():
                    sections.append(f"  - `{var_name}` → {table_name}")
            sections.append("")
            if schema_style == "compact":
                sections.append(schema.to_compact_string())
            else:
                sections.append(schema.to_prompt_string(
                    include_samples=schema_style == "full",
                ))

            if include_rows and head_rows:
                sections.append("\n## SAMPLE ROWS (optional reference)")
                for table_name, rows_str in head_rows.items():
                    sections.extend([f"\n### {table_name}", rows_str])

            if question_info:
                sections.append("\n## QUESTION ANALYSIS")
                sections.append(f"Expected answer type: {question_info.expected_type}")
                if question_info.aggregation:
                    sections.append(f"Aggregation: {question_info.aggregation}")
                if question_info.ranking_direction:
                    sections.append(
                        f"Ranking: {question_info.ranking_direction}, "
                        f"k={question_info.ranking_k}, "
                        f"criterion={question_info.ranking_criterion}"
                    )
                if question_info.time_expressions:
                    sections.append("Time ranges:")
                    for expr, start, end in question_info.time_expressions:
                        sections.append(f"  - '{expr}' → {start} to {end}")
                if question_info.is_comparison:
                    sections.append("This is a comparison question.")
            if completeness_context:
                sections.append("\n## QUESTION COMPLETENESS")
                sections.append(
                    "Use this completeness assessment. Preserve its assumptions "
                    "and satisfy every listed requirement."
                )
                sections.append(json.dumps(
                    completeness_context,
                    ensure_ascii=False,
                    indent=2,
                ))
            if query_plan:
                sections.append("\n## STRUCTURED QUERY PLAN")
                sections.append(
                    "Implement this plan exactly. Do not invent tables, columns, "
                    "filters, or placeholder values. If the plan is fallback, "
                    "re-check the available schema before answering."
                )
                sections.append(json.dumps(query_plan, ensure_ascii=False, indent=2))

            if include_examples and selected_examples:
                sections.append(
                    "\n## EXAMPLES\n"
                    "These examples illustrate calculation patterns only. Their table "
                    "variables and columns are not available unless listed in AVAILABLE DATA."
                )
                for index, example in enumerate(selected_examples, 1):
                    sections.extend([
                        f"\n### Example {index}",
                        f"Question: {example.get('question', '')}",
                        f"Code:\n```python\n{example.get('code', '')}\n```",
                        f"Answer: {example.get('answer', '')}",
                    ])

            sections.extend([
                "\n## YOUR TASK",
                f"as_of_date = \"{as_of_date}\"",
                f"Question: {question}",
                "",
                "Write Python code to answer this question. "
                "Assign the answer to `result` and the type to `answer_type`.",
                "```python",
            ])
            return "\n".join(sections)

        strategies = [
            (True, True, True, "full", 2000, 1500),
            (False, True, True, "full", 2000, 1500),
            (False, False, True, "full", 2000, 1500),
            (False, False, True, "without_samples", 2000, 1500),
            (False, False, True, "compact", 1200, 750),
            (False, False, True, "compact", 600, 300),
            (False, False, False, "compact", 600, 0),
            (False, False, False, "compact", 0, 0),
        ]
        if token_counter is None or max_tokens is None:
            return render(*strategies[0])

        token_count = 0
        for strategy in strategies:
            prompt = render(*strategy)
            token_count = token_counter(prompt)
            if token_count <= max_tokens:
                logger.info("Prompt token count: %d/%d", token_count, max_tokens)
                if strategy != strategies[0]:
                    logger.info(
                        "Compacted prompt (schema=%s, rows=%s, examples=%s)",
                        strategy[3],
                        strategy[0],
                        strategy[1],
                    )
                return prompt

        raise PromptTokenLimitError(
            f"Core prompt exceeds the configured input budget of {max_tokens} tokens "
            f"even after compaction (minimum measured: {token_count} tokens)."
        )

    def build_retry_prompt(
        self,
        original_prompt: str,
        generated_code: str,
        error_message: str,
        available_context: Optional[str] = None,
        token_counter: Optional[Callable[[str], int]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Build a retry prompt when the previous code execution failed.

        Parameters
        ----------
        original_prompt : str
            The original prompt.
        generated_code : str
            The code that was generated and failed.
        error_message : str
            The             Error message from execution.
        available_context : str, optional
            Concise variable or column information to help correct the failure.
        token_counter : Callable[[str], int], optional
            Counts tokens using the generation model's tokenizer.
        max_tokens : int, optional
            Maximum input tokens allowed for the retry prompt.

        Returns
        -------
        str
            The retry prompt.
        """
        if (token_counter is None) != (max_tokens is None):
            raise ValueError("token_counter and max_tokens must be provided together.")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be greater than zero.")

        def remove_section(prompt: str, heading: str) -> str:
            start = prompt.find(f"\n## {heading}")
            if start < 0:
                return prompt
            next_section = prompt.find("\n## ", start + 1)
            if next_section < 0:
                return prompt[:start]
            return prompt[:start] + prompt[next_section:]

        base_prompts = [original_prompt]
        compact_prompt = original_prompt
        for heading in ("SAMPLE ROWS", "EXAMPLES", "BUSINESS CONTEXT"):
            compact_prompt = remove_section(compact_prompt, heading)
            if compact_prompt not in base_prompts:
                base_prompts.append(compact_prompt)
        question_line = ""
        question_marker = "Question:"
        if question_marker in original_prompt:
            question_line = original_prompt.rsplit(question_marker, 1)[-1].splitlines()[0].strip()
        available_section = ""
        available_marker = "\n## AVAILABLE DATA"
        if available_marker in original_prompt:
            available_section = original_prompt.split(available_marker, 1)[1]
            next_section = available_section.find("\n## ")
            if next_section >= 0:
                available_section = available_section[:next_section]
        minimal_base = (
            "You are an expert Vietnamese data analyst. Use only the available "
            "DataFrames and columns. Preserve the question's exact conditions.\n"
            f"Question: {question_line}\n"
            f"## AVAILABLE DATA{available_section}"
        )
        base_prompts.append(minimal_base)

        def shorten(text: str, limit: Optional[int]) -> str:
            if limit is None or len(text) <= limit:
                return text
            if limit <= 3:
                return text[:limit]
            head = (limit - 3) // 2
            tail = limit - 3 - head
            return text[:head] + "..." + text[-tail:]

        code_limits = (None, 2000, 1000, 500, 250, 100)
        error_limits = (None, 1000, 500, 250, 150, 75)
        context_limits = (None, 1500, 750, 300, 150, 0)
        for base in base_prompts:
            for context_limit in context_limits:
                if available_context is None:
                    context = ""
                else:
                    context = shorten(available_context, context_limit)
                for code_limit in code_limits:
                    code = shorten(generated_code, code_limit)
                    for error_limit in error_limits:
                        error = shorten(error_message, error_limit)
                        sections = [base]
                        if context:
                            sections.extend(["\n## AVAILABLE VARIABLES AND COLUMNS", context])
                        sections.extend([
                            "\n## PREVIOUS ATTEMPT (FAILED)",
                            f"```python\n{code}\n```",
                            "\n## ERROR",
                            error,
                            "\n## INSTRUCTIONS",
                            "Fix the error. Use only the listed DataFrame variables and "
                            "real column names. Assign `result` and `answer_type`.",
                            "```python",
                        ])
                        retry = "\n".join(sections)
                        if token_counter is None or max_tokens is None:
                            return retry
                        token_count = token_counter(retry)
                        if token_count <= max_tokens:
                            logger.info(
                                "Retry prompt token count: %d/%d",
                                token_count,
                                max_tokens,
                            )
                            return retry

        if token_counter is not None and max_tokens is not None:
            raise PromptTokenLimitError(
                f"Retry prompt exceeds the configured input budget of {max_tokens} tokens "
                "after removing optional context and truncating error details."
            )
        return "\n".join([
            original_prompt,
            "\n## PREVIOUS ATTEMPT (FAILED)",
            f"```python\n{generated_code}\n```",
            "\n## ERROR",
            error_message,
            "\n## INSTRUCTIONS",
            "Fix the error. Assign `result` and `answer_type`.",
            "```python",
        ])
