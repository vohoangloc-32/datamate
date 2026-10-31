"""
logger.py — Per-Question Logging for DataMate

Writes detailed per-question logs to /result/logs/ for debugging,
reproducibility verification, and organizer inspection.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class QuestionLogger:
    """Logs detailed information for each question processed."""

    def __init__(self, log_dir: str):
        """
        Parameters
        ----------
        log_dir : str
            Directory to write per-question logs (e.g., /result/logs/).
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._start_times: Dict[str, float] = {}
        self._summary_path = self.log_dir / "summary.jsonl"

    def start_question(self, question_id: str):
        """Record the start time for a question."""
        self._start_times[question_id] = time.time()

    def log_question(
        self,
        question_id: str,
        question_text: str,
        business_id: str,
        final_answer: Dict[str, Any],
        generated_code: str = "",
        execution_output: str = "",
        retries: int = 0,
        error_messages: Optional[list] = None,
        extra: Optional[Dict] = None,
    ):
        """
        Write a detailed log for a single question.

        Parameters
        ----------
        question_id : str
            The question ID.
        question_text : str
            The original question text.
        business_id : str
            The business ID.
        final_answer : Dict[str, Any]
            The final formatted answer.
        generated_code : str
            The generated code (last attempt).
        execution_output : str
            stdout/stderr from code execution.
        retries : int
            Number of retries used.
        error_messages : list, optional
            Error messages from failed attempts.
        extra : Dict, optional
            Any additional metadata.
        """
        elapsed = 0.0
        if question_id in self._start_times:
            elapsed = time.time() - self._start_times.pop(question_id)

        log_entry = {
            "question_id": question_id,
            "business_id": business_id,
            "question": question_text,
            "answer": final_answer,
            "elapsed_seconds": round(elapsed, 2),
            "retries": retries,
            "timestamp": datetime.now().isoformat(),
        }

        if generated_code:
            log_entry["generated_code"] = generated_code
        if execution_output:
            log_entry["execution_output"] = execution_output[:2000]
        if error_messages:
            log_entry["errors"] = error_messages
        if extra:
            log_entry["extra"] = extra

        # Write per-question log file
        question_log_path = self.log_dir / f"{question_id}.json"
        try:
            with open(question_log_path, "w", encoding="utf-8") as f:
                json.dump(log_entry, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Failed to write log for %s: %s", question_id, e)

        # Append to summary
        summary_line = {
            "id": question_id,
            "type": final_answer.get("type", "abstain"),
            "answer": final_answer.get("answer"),
            "elapsed": round(elapsed, 2),
            "retries": retries,
        }
        try:
            with open(self._summary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary_line, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error("Failed to write summary for %s: %s", question_id, e)

        logger.info("Logged question %s: type=%s, elapsed=%.1fs, retries=%d",
                     question_id, final_answer.get("type"), elapsed, retries)
