# Implementation Plan: Ranking and Abstention Calibration

**Branch**: `001-ranking-abstention-calibration` | **Date**: 2026-03-19 | **Spec**: [spec.md](./spec.md)

## Summary

Validate and, where required, refine the existing repeated-candidate pipeline so verifier evidence dominates candidate selection, confidence floors prevent unsupported answers, semantic `unanswerable` remains distinct from `abstain`, and bounded notebook runs produce comparable quality metrics.

## Technical Context

**Language/Version**: Python 3.x

**Primary Dependencies**: pytest, pandas, vLLM, Qwen2.5 GPTQ, Jupyter

**Storage**: JSONL, CSV, Markdown, notebook outputs, and local logs

**Testing**: pytest, Python compilation, notebook compilation, archive synchronization checks

**Target Platform**: Local CPU test environment and Kaggle GPU runtime

**Project Type**: Python inference pipeline with Kaggle notebooks

**Performance Goals**: Bounded smoke runs finish early; full runs remain configurable by sample size and candidate count.

**Constraints**: Preserve answer schema, avoid silent fallbacks, and do not claim GPU accuracy without a completed GPU evaluation.

## Constitution Check

- Correctness Before Coverage: pass; acceptance tests cover both abstention and semantic unanswerable.
- Evidence-Driven Inference: pass; ranking uses verifier, execution, type, retry, and consensus evidence.
- Reproducible Evaluation: pass; configuration and required metrics are part of the evaluation report.
- Safe Integration: pass; source/notebook/archive synchronization is a validation gate.
- Focused Simplicity: pass; reuse existing confidence gate and notebook pipeline.

## Project Structure

```text
src/
├── predict.py
└── modules/
    ├── confidence_gate.py
    ├── answer_formatter.py
    ├── code_executor.py
    └── prompt_builder.py
tests/
└── test_confidence_gate.py
specs/001-ranking-abstention-calibration/
├── spec.md
├── plan.md
└── tasks.md
kaggle_train_baseline.ipynb
datamate_code.zip
```

**Structure Decision**: Keep the existing single Python pipeline and add only focused tests/reporting hooks around the current confidence and notebook surfaces.

## Implementation Phases

1. Establish baseline behavior with focused tests and a bounded deterministic candidate matrix.
2. Verify candidate ranking, confidence floors, answer-type preservation, and failure diagnostics.
3. Verify notebook/source/archive synchronization and notebook compilation.
4. Execute a small Kaggle smoke run, then the configured 500-question evaluation if the smoke run passes.
5. Compare metrics against the baseline and document any threshold decision.

## Complexity Tracking

No constitution violations are expected. A separate model or database is intentionally out of scope.
