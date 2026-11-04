"""
predict.py — Main Orchestrator for DataMate

End-to-end pipeline: reads questions, loads data, generates code,
executes it, formats answers, and writes submission.jsonl.

Usage:
    python predict.py --data-dir /data --output-dir /result --model-dir /app/models/qwen2.5-coder-14b
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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
from modules.retry_engine import RetryEngine
from modules.answer_formatter import format_answer, build_submission_line
from modules.confidence_gate import ConfidenceGate
from modules.query_planner import build_query_plan
from modules.question_logger import QuestionLogger

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

        # Split the model across two GPUs by default; override with VLLM_TP when needed.
        tp_size = int(os.environ.get("VLLM_TP", "2"))
        max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
        max_new_tokens = int(os.environ.get("VLLM_MAX_NEW_TOKENS", "1024"))
        temperature = float(os.environ.get("DATAMATE_TEMPERATURE", "0.7"))
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("DATAMATE_TEMPERATURE must be between 0 and 2.")
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
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop=["```\n", "\n\n\n"],
            include_stop_str_in_output=True,  # keep closing ``` for code extraction
        )

        def generate(prompt: str) -> str:
            text = _format_chat_prompt(vllm_tokenizer, prompt)
            outputs = llm.generate([text], sampling_params)
            return outputs[0].outputs[0].text

        logger.info(
            "Loaded vLLM model from %s (tensor_parallel_size=%d, temperature=%.2f)",
            model_dir,
            tp_size,
            temperature,
        )
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
        confidence_threshold: float = 0.2,
        confidence_runs: int = 4,
        adaptive_confidence: bool = True,
        min_confidence_runs: int = 2,
        question_completeness_threshold: float = 0.85,
        llm_generate_fn: Optional[Callable[[str], str]] = None,
        prompt_token_counter: Optional[Callable[[str], int]] = None,
        max_prompt_tokens: Optional[int] = None,
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

        self.max_retries = max_retries
        if confidence_runs < 1:
            raise ValueError("confidence_runs must be greater than zero.")
        if min_confidence_runs < 1 or min_confidence_runs > confidence_runs:
            raise ValueError(
                "min_confidence_runs must be between 1 and confidence_runs."
            )
        if not 0.0 < question_completeness_threshold <= 1.0:
            raise ValueError("question_completeness_threshold must be between 0 and 1.")
        self.confidence_runs = confidence_runs
        self.adaptive_confidence = adaptive_confidence
        self.min_confidence_runs = min_confidence_runs
        self.question_completeness_threshold = question_completeness_threshold

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
            # 1. Load business data
            self._load_business(biz_id)
            tables = self._data_cache[biz_id]
            schema = self._schema_cache[biz_id]
            readme = self._readme_cache[biz_id]

            # Load the model tokenizer before building a prompt so the prompt
            # budget is measured with the same tokenizer used for generation.
            retry_engine = self._ensure_llm_loaded()

            # 2. Analyze question
            as_of = date.fromisoformat(as_of_date)
            q_info = self.question_analyzer.analyze(question, as_of)

            completeness = self.confidence_gate.evaluate_question_completeness(
                self._llm_generate_fn,
                question=question,
                as_of_date=as_of_date,
                question_info=q_info,
                schema=schema.to_compact_string(),
                rulebook=self.rulebook.to_prompt_string(
                    relevant_only=question,
                    max_length=3000,
                ),
                business_context=readme.to_prompt_string(max_length=2000),
            )
            query_plan = build_query_plan(
                self._llm_generate_fn,
                question=question,
                as_of_date=as_of_date,
                schema=schema.to_compact_string(),
                rulebook=self.rulebook.to_prompt_string(
                    relevant_only=question,
                    max_length=3000,
                ),
                tables=tables,
                table_var_names=self.code_executor.get_table_var_mapping(tables),
                expected_type=q_info.expected_type,
            )
            if (
                completeness["question_status"] == "unanswerable"
                and completeness["confidence"] >= self.question_completeness_threshold
                and completeness.get("missing_context")
            ):
                fallback = {
                    "type": "abstain",
                    "reason": (
                        "question_incomplete: "
                        + str(completeness.get("reason", "missing required context"))
                    ),
                }
                self.question_logger.log_question(
                    question_id=q_id,
                    question_text=question,
                    business_id=biz_id,
                    final_answer=fallback,
                    error_messages=[
                        "missing_context="
                        + json.dumps(completeness.get("missing_context", []), ensure_ascii=False)
                    ],
                )
                return fallback

            # 3. Build prompt
            table_vars = self.code_executor.get_table_var_mapping(tables)
            head_rows = self.code_executor.get_head_rows(tables, n=3)

            prompt = self.prompt_builder.build(
                question=question,
                as_of_date=as_of_date,
                schema=schema,
                readme_context=readme,
                rulebook=self.rulebook,
                question_info=q_info,
                table_var_names=table_vars,
                head_rows=head_rows,
                completeness_context=completeness,
                query_plan=query_plan,
                token_counter=self._prompt_token_counter,
                max_tokens=(
                    max(1, self._max_prompt_tokens - min(1024, self._max_prompt_tokens // 5))
                    if self._prompt_token_counter is not None
                    and self._max_prompt_tokens is not None
                    else None
                ),
            )

            # 4. Generate and execute repeated independent runs with retries.
            run_answers = []
            candidate_records = []
            failed_runs = 0
            total_retries = 0
            run_errors = []
            last_result = None
            for run_index in range(self.confidence_runs):
                exec_result, retries = retry_engine.run(
                    prompt=prompt,
                    tables=tables,
                    as_of_date=as_of_date,
                    expected_type=q_info.expected_type,
                    token_counter=self._prompt_token_counter,
                    max_prompt_tokens=self._max_prompt_tokens,
                )
                last_result = exec_result
                total_retries += retries
                if not exec_result.success:
                    failed_runs += 1
                if exec_result.error:
                    run_errors.append(exec_result.error)
                run_answers.append(
                    format_answer(
                        value=exec_result.result,
                        answer_type=exec_result.answer_type,
                    )
                )
                candidate_records.append(
                    {
                        "answer": run_answers[-1],
                        "execution_success": exec_result.success,
                        "execution_output": exec_result.stdout,
                        "code": exec_result.code,
                        "retries": retries,
                    }
                )
                if (
                    self.adaptive_confidence
                    and run_index + 1 >= self.min_confidence_runs
                    and run_answers[-1].get("type") not in ("abstain", "unanswerable")
                    and all(answer == run_answers[-1] for answer in run_answers)
                    and total_retries == 0
                    and failed_runs == 0
                    and completeness.get("question_status") == "answerable"
                    and float(completeness.get("confidence", 0.0)) >= 0.8
                    and q_info.confidence >= 0.75
                ):
                    logger.info(
                        "Stopping repeated runs early for %s after %d matching runs",
                        q_id,
                        run_index + 1,
                    )
                    break

            use_post_verifier = not (
                self.adaptive_confidence
                and len(run_answers) < self.confidence_runs
                and len(run_answers) >= self.min_confidence_runs
                and all(answer == run_answers[0] for answer in run_answers)
                and failed_runs == 0
                and total_retries == 0
            )

            # 5. Select the most common answer and evaluate run consistency.
            final_answer = self.confidence_gate.evaluate_runs(
                run_answers,
                candidate_records=candidate_records,
                failed_runs=failed_runs,
                execution_retries=total_retries,
                question_info=q_info,
                completeness_context=completeness,
                prompt_verifier_fn=self._llm_generate_fn if use_post_verifier else None,
                verifier_context={
                    "question": question,
                    "as_of_date": as_of_date,
                    "expected_type": q_info.expected_type,
                    "schema": schema.to_compact_string(),
                    "rulebook": self.rulebook.to_prompt_string(
                        relevant_only=question,
                        max_length=3000,
                    ),
                    "question_completeness": completeness,
                },
            )
            exec_result = last_result
            retries = total_retries

            # 7. Log
            self.question_logger.log_question(
                question_id=q_id,
                question_text=question,
                business_id=biz_id,
                final_answer=final_answer,
                generated_code=exec_result.code if exec_result else "",
                execution_output=exec_result.stdout,
                retries=retries,
                error_messages=run_errors or None,
                extra={"query_plan": query_plan, "question_analysis": q_info.to_dict()},
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
        finally:
            # Release prompt, retry, and candidate objects before the next question.
            # Business tables remain in the intentional per-business cache.
            gc.collect()

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
    parser.add_argument("--confidence-threshold", type=float, default=0.3,
                        help="Minimum confidence to submit an answer (0-1)")
    parser.add_argument("--confidence-runs", type=int, default=4,
                        help="Number of independent runs used to assess confidence")
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
        confidence_runs=args.confidence_runs,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
