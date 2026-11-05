"""
predict.py — Main Orchestrator for DataMate

End-to-end pipeline: reads questions, loads data, generates code,
executes it, formats answers, and writes submission.jsonl.

Usage:
    python predict.py --data-dir /data --output-dir /result --model-dir /app/models/qwen2.5-coder-14b
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.data_loader import load_business_data, clear_cache
from modules.schema_registry import SchemaRegistry
from modules.readme_parser import parse_readme
from modules.rulebook_interpreter import parse_rulebook
from modules.time_resolver import LunarCalendar
from modules.question_understanding import QuestionAnalyzer
from modules.prompt_builder import PromptBuilder
from modules.code_executor import CodeExecutor
from modules.analysis_plan import (
    AnalysisPlan,
    AnalysisPlanError,
    execute_analysis_plan,
)
from modules.retry_engine import RetryEngine
from modules.answer_formatter import format_answer, build_submission_line
from modules.confidence_gate import ConfidenceGate
from modules.question_logger import QuestionLogger
from modules.value_grounder import ValueGrounder
from modules.accounting_cleaner import (
    format_accounting_context,
    prepare_tables_for_question,
)

logger = logging.getLogger(__name__)


def _format_chat_prompt(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


class _TokenCountingGenerateFn:
    """Callable model adapter that exposes the exact tokenizer input budget."""

    def __init__(self, generate_fn, tokenizer, max_prompt_tokens: int):
        self._generate_fn = generate_fn
        self._tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens

    def __call__(self, prompt: str) -> str:
        return self._generate_fn(prompt)

    def count_tokens(self, prompt: str) -> int:
        formatted_prompt = _format_chat_prompt(self._tokenizer, prompt)
        return len(self._tokenizer.encode(formatted_prompt, add_special_tokens=False))


# ---------------------------------------------------------------------------
# LLM interface
# ---------------------------------------------------------------------------

def create_llm_generate_fn(model_dir: str, device: str = "cuda") -> Callable[[str], str]:
    """
    Create the LLM generation function using vLLM.

    Parameters
    ----------
    model_dir : str
        Path to the model weights.
    device : str
        Device to use ('cuda', 'cpu').

    Returns
    -------
    Callable[[str], str]
        Function that takes a prompt and returns the generated text.
    """
    try:
        from vllm import LLM, SamplingParams  # type: ignore  # optional dependency

        # Number of GPUs used for tensor parallelism. A 3B model fits on a single
        # 16 GB T4, so default to 1 (more robust); set VLLM_TP=2 to use both GPUs.
        tp_size = int(os.environ.get("VLLM_TP", "1"))
        max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
        max_new_tokens = int(os.environ.get("VLLM_MAX_NEW_TOKENS", "1024"))
        prompt_safety_tokens = int(os.environ.get("DATAMATE_PROMPT_SAFETY_TOKENS", "512"))
        max_prompt_tokens = max_model_len - max_new_tokens - prompt_safety_tokens
        if max_prompt_tokens < 1:
            raise ValueError(
                "VLLM_MAX_MODEL_LEN must exceed VLLM_MAX_NEW_TOKENS plus "
                "DATAMATE_PROMPT_SAFETY_TOKENS."
            )

        llm = LLM(
            model=model_dir,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.85,
            dtype="float16",  # T4 (compute capability 7.5) has no bf16 support
        )
        vllm_tokenizer = llm.get_tokenizer()
        sampling_params = SamplingParams(
            temperature=0.0,  # Greedy for reproducibility
            max_tokens=max_new_tokens,
            stop=["```\n", "\n\n\n"],
            include_stop_str_in_output=True,  # keep closing ``` for code extraction
        )

        def generate(prompt: str) -> str:
            text = _format_chat_prompt(vllm_tokenizer, prompt)
            outputs = llm.generate([text], sampling_params)
            return outputs[0].outputs[0].text

        logger.info("Loaded vLLM model from %s (tensor_parallel_size=%d)",
                    model_dir, tp_size)
        return _TokenCountingGenerateFn(generate, vllm_tokenizer, max_prompt_tokens)

    except Exception as e:  # vLLM missing (ImportError) or failed to initialise
        logger.warning("vLLM not available (%s: %s), falling back to transformers",
                       type(e).__name__, e)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        from transformers import StoppingCriteria, StoppingCriteriaList

        class _CodeBlockClosed(StoppingCriteria):
            """Stop once the model has opened AND closed a ``` code block."""

            def __init__(self, prompt_len: int):
                self.prompt_len = prompt_len

            def __call__(self, input_ids, scores, **kwargs):
                text = tokenizer.decode(input_ids[0][self.prompt_len:],
                                        skip_special_tokens=True)
                return text.count("```") >= 2

        max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
        max_new_tokens = int(os.environ.get("VLLM_MAX_NEW_TOKENS", "1024"))
        prompt_safety_tokens = int(os.environ.get("DATAMATE_PROMPT_SAFETY_TOKENS", "512"))
        max_prompt_tokens = max_model_len - max_new_tokens - prompt_safety_tokens
        if max_prompt_tokens < 1:
            raise ValueError(
                "VLLM_MAX_MODEL_LEN must exceed VLLM_MAX_NEW_TOKENS plus "
                "DATAMATE_PROMPT_SAFETY_TOKENS."
            )

        def generate(prompt: str) -> str:
            # Qwen-Instruct expects its chat format; raw text makes the model
            # ramble on until max_new_tokens.
            text = _format_chat_prompt(tokenizer, prompt)
            inputs = tokenizer(text, return_tensors="pt",
                               add_special_tokens=False).to(model.device)
            prompt_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([_CodeBlockClosed(prompt_len)]),
                )
            response = tokenizer.decode(outputs[0][prompt_len:],
                                         skip_special_tokens=True)
            return response

        logger.info("Loaded transformers model from %s", model_dir)
        return _TokenCountingGenerateFn(generate, tokenizer, max_prompt_tokens)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class DataMatePipeline:
    """Main orchestrator for the DataMate agent."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        model_dir: str = "",
        max_retries: int = 3,
        confidence_threshold: float = 0.45,
        llm_generate_fn: Optional[Callable[[str], str]] = None,
        prompt_token_counter: Optional[Callable[[str], int]] = None,
        max_prompt_tokens: Optional[int] = None,
        analysis_plan_enabled: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.model_dir = model_dir

        # Output paths
        self.submission_path = self.output_dir / "submission.jsonl"
        self.log_dir = self.output_dir / "logs"

        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Load rulebook
        rulebook_path = self.data_dir / "rulebook.md"
        self.rulebook = parse_rulebook(str(rulebook_path))
        logger.info("Loaded rulebook: %d rules", len(self.rulebook.rules))

        # Load lunar calendar
        self.calendar = LunarCalendar()
        calendar_path = self.data_dir / "calendar.csv"
        self.calendar.load(str(calendar_path))

        # Initialize components
        self.question_analyzer = QuestionAnalyzer(self.calendar)
        self.prompt_builder = PromptBuilder()
        self.code_executor = CodeExecutor(timeout_seconds=30)
        self.confidence_gate = ConfidenceGate(
            min_confidence=confidence_threshold,
        )
        self.question_logger = QuestionLogger(str(self.log_dir))

        # LLM generation function and retry engine
        self._llm_generate_fn = llm_generate_fn
        if (prompt_token_counter is None) != (max_prompt_tokens is None):
            raise ValueError(
                "prompt_token_counter and max_prompt_tokens must be provided together."
            )
        if max_prompt_tokens is not None and max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be greater than zero.")
        self._prompt_token_counter = prompt_token_counter
        self._max_prompt_tokens = max_prompt_tokens
        self._retry_engine = None

        # Data cache
        self._data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._schema_cache: Dict[str, SchemaRegistry] = {}
        self._readme_cache: Dict[str, Any] = {}
        self._grounder_cache: Dict[str, Any] = {}

        self.max_retries = max_retries
        self.analysis_plan_enabled = analysis_plan_enabled

    def _ensure_llm_loaded(self) -> RetryEngine:
        """Lazily load the LLM model if not provided."""
        if self._llm_generate_fn is None:
            logger.info("Loading LLM model from %s...", self.model_dir)
            t0 = time.time()
            self._llm_generate_fn = create_llm_generate_fn(self.model_dir)
            logger.info("LLM loaded in %.1fs", time.time() - t0)

        if self._prompt_token_counter is None:
            if isinstance(self._llm_generate_fn, _TokenCountingGenerateFn):
                self._prompt_token_counter = self._llm_generate_fn.count_tokens
        if self._max_prompt_tokens is None:
            if isinstance(self._llm_generate_fn, _TokenCountingGenerateFn):
                self._max_prompt_tokens = self._llm_generate_fn.max_prompt_tokens

        if self._retry_engine is None:
            self._retry_engine = RetryEngine(
                code_executor=self.code_executor,
                prompt_builder=self.prompt_builder,
                llm_generate_fn=self._llm_generate_fn,
                max_retries=self.max_retries,
            )
        return self._retry_engine

    def _load_business(self, business_id: str):
        """Load and cache business data, schema, and README."""
        if business_id not in self._data_cache:
            biz_dir = self.data_dir / "data" / business_id
            tables = load_business_data(str(biz_dir))
            self._data_cache[business_id] = tables

            schema = SchemaRegistry().build_from_dict(tables)
            self._schema_cache[business_id] = schema

            readme_path = biz_dir / "README.md"
            readme = parse_readme(str(readme_path))
            self._readme_cache[business_id] = readme

            self._grounder_cache[business_id] = ValueGrounder(tables)

            logger.info("Loaded business %s: %d tables", business_id, len(tables))

    def process_question(self, question_record: Dict) -> Dict[str, Any]:
        """
        Process a single question and return the formatted answer.

        Parameters
        ----------
        question_record : Dict
            A record from questions.jsonl with keys:
            "id", "business_id", "question", "as_of_date".

        Returns
        -------
        Dict[str, Any]
            The formatted answer for submission.
        """
        q_id = question_record["id"]
        biz_id = question_record["business_id"]
        question = question_record["question"]
        as_of_date = question_record["as_of_date"]

        self.question_logger.start_question(q_id)

        try:
            # 1. Load business data and apply Rulebook metric-aware filtering
            self._load_business(biz_id)
            raw_tables = self._data_cache[biz_id]
            tables = prepare_tables_for_question(raw_tables, question)
            schema = self._schema_cache[biz_id]
            readme = self._readme_cache[biz_id]
            grounder = self._grounder_cache.get(biz_id)
            grounded_parts = [
                part
                for part in (
                    grounder.format_grounded_context(question) if grounder else None,
                    format_accounting_context(raw_tables, question),
                )
                if part
            ]
            grounded_context = "\n\n".join(grounded_parts) if grounded_parts else None

            # Load the model tokenizer before building a prompt so the prompt
            # budget is measured with the same tokenizer used for generation.
            retry_engine = self._ensure_llm_loaded()

            # 2. Analyze question
            as_of = date.fromisoformat(as_of_date)
            q_info = self.question_analyzer.analyze(question, as_of)

            # 3. Build prompt
            table_vars = self.code_executor.get_table_var_mapping(tables)
            head_rows = self.code_executor.get_head_rows(tables, n=3)

            prompt_options = dict(
                question=question,
                as_of_date=as_of_date,
                schema=schema,
                readme_context=readme,
                rulebook=self.rulebook,
                question_info=q_info,
                table_var_names=table_vars,
                head_rows=head_rows,
                token_counter=self._prompt_token_counter,
                max_tokens=(
                    max(1, self._max_prompt_tokens - min(1024, self._max_prompt_tokens // 5))
                    if self._prompt_token_counter is not None
                    and self._max_prompt_tokens is not None
                    else None
                ),
                grounded_context=grounded_context,
            )

            plan_status = "disabled"
            plan_error = None
            plan_response = None
            exec_result = None
            retries = 0

            if self.analysis_plan_enabled:
                if q_info.expected_type not in {"number", "string", "list", "table"}:
                    plan_status = "skipped_unsupported_type"
                else:
                    plan_status = "fallback"
                    try:
                        plan_prompt = self.prompt_builder.build(
                            **prompt_options,
                            mode="analysis_plan",
                        )
                        generate_fn = self._llm_generate_fn
                        if generate_fn is None:
                            raise RuntimeError("LLM generator was not initialized.")
                        plan_response = generate_fn(plan_prompt)
                        plan = AnalysisPlan.from_response(
                            plan_response,
                            tables,
                            expected_type=q_info.expected_type,
                        )
                        exec_result = execute_analysis_plan(plan, tables)
                        plan_status = "executed"
                    except AnalysisPlanError as exc:
                        plan_error = str(exc)
                        logger.info(
                            "Analysis plan unsupported for question %s; using code fallback: %s",
                            q_id,
                            plan_error,
                        )
                    except Exception as exc:
                        plan_error = f"{type(exc).__name__}: {exc}"
                        logger.exception(
                            "Analysis plan failed for question %s; using code fallback",
                            q_id,
                        )

            if exec_result is None:
                prompt = self.prompt_builder.build(**prompt_options)
                exec_result, retries = retry_engine.run(
                    prompt=prompt,
                    tables=tables,
                    as_of_date=as_of_date,
                    expected_type=q_info.expected_type,
                    token_counter=self._prompt_token_counter,
                    max_prompt_tokens=self._max_prompt_tokens,
                    question=question,
                )

            # 6. Confidence gate
            final_answer = self.confidence_gate.evaluate(
                answer=format_answer(
                    value=exec_result.result,
                    answer_type=exec_result.answer_type,
                ),
                question_info=q_info,
                execution_retries=retries,
            )

            # 7. Log
            self.question_logger.log_question(
                question_id=q_id,
                question_text=question,
                business_id=biz_id,
                final_answer=final_answer,
                generated_code=exec_result.code,
                execution_output=exec_result.stdout,
                retries=retries,
                error_messages=[exec_result.error] if exec_result.error else None,
                extra={
                    "analysis_plan_status": plan_status,
                    "analysis_plan_error": plan_error,
                    "analysis_plan_response": plan_response[:4000]
                    if plan_response
                    else None,
                },
            )

            return final_answer

        except Exception as e:
            logger.error("Failed to process question %s: %s", q_id, e, exc_info=True)
            fallback = {"type": "abstain", "reason": f"pipeline_error: {str(e)}"}
            self.question_logger.log_question(
                question_id=q_id,
                question_text=question,
                business_id=biz_id,
                final_answer=fallback,
                error_messages=[str(e)],
            )
            return fallback

    def run(self):
        """Run the full pipeline on all questions."""
        # Load questions
        questions_path = self.data_dir / "questions.jsonl"
        if not questions_path.exists():
            logger.error("Questions file not found: %s", questions_path)
            return

        questions = []
        with open(questions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    questions.append(json.loads(line))

        logger.info("Loaded %d questions", len(questions))

        # Sort by business_id to maximize cache hits
        questions.sort(key=lambda q: q["business_id"])

        # Process all questions
        total_start = time.time()
        results = []

        for i, q in enumerate(questions, 1):
            logger.info("Processing question %d/%d: %s", i, len(questions), q["id"])
            answer = self.process_question(q)
            submission_line = build_submission_line(q["id"], answer)
            results.append(submission_line)

            # Write incrementally
            with open(self.submission_path, "a", encoding="utf-8") as f:
                f.write(submission_line + "\n")

            elapsed = time.time() - total_start
            avg_per_q = elapsed / i
            remaining = (len(questions) - i) * avg_per_q
            logger.info(
                "Progress: %d/%d | Elapsed: %.0fs | Avg: %.1fs/q | ETA: %.0fs",
                i, len(questions), elapsed, avg_per_q, remaining
            )

        total_elapsed = time.time() - total_start
        logger.info("Pipeline complete: %d questions in %.0fs (avg %.1fs/q)",
                     len(questions), total_elapsed, total_elapsed / max(len(questions), 1))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DataMate - Vietnamese QA over Tabular Data")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to the data directory (contains data/, questions.jsonl, etc.)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Path to the output directory (writes submission.jsonl and logs/)")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Path to the LLM model weights")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Maximum retries per question")
    parser.add_argument("--confidence-threshold", type=float, default=0.45,
                        help="Minimum confidence to submit an answer (0-1)")
    parser.add_argument(
        "--analysis-plan",
        action="store_true",
        help="Try a validated structured analysis plan before pandas code fallback",
    )
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level")

    args = parser.parse_args()

    # Ensure output directory exists before configuring file handler
    os.makedirs(args.output_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                os.path.join(args.output_dir, "predict.log"),
                encoding="utf-8",
            ),
        ],
    )

    # Clear any stale submission file
    submission_path = os.path.join(args.output_dir, "submission.jsonl")
    if os.path.exists(submission_path):
        os.remove(submission_path)

    # Run pipeline
    pipeline = DataMatePipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        max_retries=args.max_retries,
        confidence_threshold=args.confidence_threshold,
        analysis_plan_enabled=args.analysis_plan,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
