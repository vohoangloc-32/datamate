# DataMate Constitution

## Core Principles

### I. Correctness Before Coverage
Answers must preserve the distinction between answerable, unanswerable, and abstained results. Changes must be evaluated against measurable answer quality and must not trade correctness for coverage without an explicit threshold decision.

### II. Evidence-Driven Inference
Candidate selection and confidence decisions must use observable execution, type, verifier, and consensus evidence. Heuristics must be deterministic where possible and must expose failures rather than silently converting them into successful answers.

### III. Reproducible Evaluation
Every inference change must have focused tests and a reproducible evaluation path. Reports must identify the sample, configuration, model, and metrics used; local tests must not be presented as proof of GPU benchmark accuracy.

### IV. Safe Integration
Source modules, prompts, notebooks, and packaged archives must remain synchronized when they represent the same runtime. Changes must preserve existing output contracts unless a documented migration is included.

### V. Focused Simplicity
Prefer the smallest change that addresses the measured failure. Do not add a new model, data store, or orchestration layer when existing generation, execution, verifier, and reporting components can provide the required evidence.

## Quality Gates

- Python changes pass the focused pytest suite and syntax checks.
- Notebook changes compile and are checked against the source configuration.
- Evaluation reports include accuracy, coverage, abstain rate, execution success, and candidate agreement when those values are available.
- Failures, missing inputs, and unsupported environments are reported explicitly.

## Governance
This constitution governs inference and evaluation changes. A plan may document a justified exception, but it must name the violated principle, explain why the simpler alternative is insufficient, and add a validation or migration step.

**Version**: 1.0.0 | **Ratified**: 2026-03-19 | **Last Amended**: 2026-03-19
