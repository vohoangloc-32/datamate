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

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from .question_understanding import QuestionInfo
from .readme_parser import BusinessContext
from .rulebook_interpreter import Rulebook
from .schema_registry import SchemaRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Vietnamese data analyst. Your task is to answer a question about a business's data by writing Python pandas code.

IMPORTANT RULES:
1. The data is already loaded as pandas DataFrames in variables. Use the exact variable names provided.
2. Write Python code that computes the answer and assigns it to a variable called `result`.
3. `result` must be a single value matching the expected answer type.
4. If the data does not contain enough information to answer, set `result = None` and `answer_type = "unanswerable"`.
5. Apply the business rules provided below (e.g., how to compute revenue, what counts as a "week", etc.).
6. Handle edge cases: empty DataFrames, missing values, division by zero.
7. For date filtering, use the as_of_date to resolve relative time expressions.
8. Do NOT use print() — just assign to `result`.
9. For string answers, return the exact name as it appears in the data.
10. For numeric answers, return an integer if exact, or float rounded to 2 decimals.
11. For percentage answers, return the numeric value (e.g., 12.45 not 0.1245).
12. For date answers, return ISO format string "YYYY-MM-DD" or "YYYY-MM".
13. For boolean answers, return True or False.
14. For list answers, return a Python list in the order requested.
15. For table answers, return a dict with "columns" (list of strings) and "rows" (list of lists).
16. Do NOT write any import statements. `pd`, `np`, `math`, `re`, `datetime`, `dt`, `date` and `timedelta` are already available.
17. Define every variable before using it; the code must be complete and self-contained.

OUTPUT FORMAT:
Your response must be a Python code block that:
- Uses the available DataFrame variables
- Computes the answer
- Assigns `result` = the answer value
- Assigns `answer_type` = one of: "number", "percent", "string", "date", "boolean", "list", "table", "unanswerable"
"""


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

        Returns
        -------
        str
            The complete prompt string.
        """
        sections = []

        # --- System ---
        sections.append(SYSTEM_PROMPT)

        # --- Business Context ---
        if readme_context and readme_context.raw_text:
            sections.append("## BUSINESS CONTEXT")
            sections.append(readme_context.to_prompt_string(max_length=1500))

        # --- Rulebook ---
        if rulebook and rulebook.rules:
            sections.append("\n## BUSINESS RULES")
            if question_info:
                sections.append(rulebook.to_prompt_string(
                    relevant_only=question, max_length=2000
                ))
            else:
                sections.append(rulebook.to_prompt_string(max_length=2000))

        # --- Data Schema ---
        sections.append("\n## AVAILABLE DATA")
        if table_var_names:
            sections.append("Variables available in your code:")
            for table_name, var_name in table_var_names.items():
                sections.append(f"  - `{var_name}` → {table_name}")
        sections.append("")
        sections.append(schema.to_prompt_string(include_samples=True))

        # --- Sample rows ---
        if head_rows:
            sections.append("\n## SAMPLE ROWS (first 3 rows of each table)")
            for table_name, rows_str in head_rows.items():
                sections.append(f"\n### {table_name}")
                sections.append(rows_str)

        # --- Question info ---
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

        # --- Few-shot examples ---
        if self.few_shot_examples:
            sections.append("\n## EXAMPLES")
            selected = []
            if question_info and question_info.expected_type:
                for ex in self.few_shot_examples:
                    if ex.get("expected_type") == question_info.expected_type:
                        selected.append(ex)
            for ex in self.few_shot_examples:
                if len(selected) >= 2:
                    break
                if ex not in selected:
                    selected.append(ex)

            for i, ex in enumerate(selected[:2], 1):
                sections.append(f"\n### Example {i}")
                sections.append(f"Question: {ex.get('question', '')}")
                sections.append(f"Code:\n```python\n{ex.get('code', '')}\n```")
                sections.append(f"Answer: {ex.get('answer', '')}")

        # --- The actual question ---
        sections.append("\n## YOUR TASK")
        sections.append(f"as_of_date = \"{as_of_date}\"")
        sections.append(f"Question: {question}")
        sections.append("")
        sections.append(
            "Write Python code to answer this question. "
            "Assign the answer to `result` and the type to `answer_type`."
        )
        sections.append("```python")

        return "\n".join(sections)

    def build_retry_prompt(
        self,
        original_prompt: str,
        generated_code: str,
        error_message: str,
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
            The error message from execution.

        Returns
        -------
        str
            The retry prompt.
        """
        retry = [
            original_prompt,
            f"\n\n## PREVIOUS ATTEMPT (FAILED)",
            f"```python\n{generated_code}\n```",
            f"\n## ERROR",
            f"{error_message}",
            f"\n## INSTRUCTIONS",
            "Fix the error in the code above. Assign the answer to `result` "
            "and the type to `answer_type`.",
            "```python",
        ]
        return "\n".join(retry)
