# Feature Specification: Ranking and Abstention Calibration

**Feature Branch**: `001-ranking-abstention-calibration`

**Created**: 2026-03-19

**Status**: Draft

**Input**: Calibrate candidate ranking and abstention using measurable evaluation metrics.

## User Scenarios & Testing

### User Story 1 - Select the best candidate (Priority: P1)

As an inference operator, I need the pipeline to select the semantically best candidate rather than merely the most frequent candidate.

**Why this priority**: Incorrect selection directly lowers answer accuracy.

**Independent Test**: Provide candidates with different verifier confidence, execution status, expected-type match, and consensus; verify the highest-quality candidate wins.

**Acceptance Scenarios**:

1. **Given** two executable candidates with different verifier confidence, **When** they are ranked, **Then** the candidate with stronger verifier evidence wins unless it violates a hard safety floor.
2. **Given** no verifier output, **When** candidates are ranked, **Then** the existing consensus fallback remains deterministic.

### User Story 2 - Abstain only when evidence is insufficient (Priority: P1)

As an evaluator, I need low-confidence or failed candidate sets to abstain without converting valid semantic `unanswerable` results into a different answer type.

**Why this priority**: Unnecessary abstention harms coverage, while unsafe guessing harms correctness.

**Independent Test**: Exercise low-confidence, failed-execution, disagreement, incomplete-question, and explicit-unanswerable cases.

**Acceptance Scenarios**:

1. **Given** verifier confidence below the configured floor, **When** a result is finalized, **Then** the result is `abstain`.
2. **Given** a valid candidate declaring the question unanswerable, **When** confidence is sufficient, **Then** `unanswerable` is preserved.

### User Story 3 - Measure the trade-off (Priority: P2)

As an operator, I need a repeatable report of quality and coverage before changing thresholds or deploying a notebook.

**Why this priority**: Local unit tests cannot establish real model accuracy.

**Independent Test**: Run a bounded sample and verify the report contains all available required metrics.

**Acceptance Scenarios**:

1. **Given** a bounded evaluation run, **When** it finishes, **Then** it reports accuracy, coverage, abstain rate, execution success, and candidate agreement.
2. **Given** a missing model or dataset input, **When** evaluation starts, **Then** it fails explicitly with an actionable diagnostic.

## Edge Cases

- All candidates are distinct and none reaches the confidence threshold.
- Candidates mix answer types or contain malformed verifier output.
- A candidate executes successfully but returns an implausible value.
- The question completeness verifier reports ambiguous or missing context.
- Notebook runtime has fewer GPUs or a different CUDA version than expected.

## Requirements

### Functional Requirements

- **FR-001**: The system MUST rank candidates using verifier evidence when it is available.
- **FR-002**: The system MUST retain a deterministic consensus fallback when verifier evidence is unavailable.
- **FR-003**: The system MUST enforce a configurable low-confidence floor that cannot be rescued by metadata bonuses.
- **FR-004**: The system MUST distinguish `abstain` from semantic `unanswerable`.
- **FR-005**: The system MUST record candidate execution success, retry count, answer type, and selected-candidate evidence.
- **FR-006**: The evaluation path MUST report accuracy, coverage, abstain rate, execution success, and candidate agreement for available ground truth.
- **FR-007**: Missing inputs and incompatible runtime requirements MUST produce explicit errors.
- **FR-008**: Existing source, prompt, notebook, and archive interfaces MUST remain compatible.

### Key Entities

- **Candidate**: Generated code, execution result, answer type, verifier score, retry count, and evidence used for ranking.
- **Confidence decision**: Selected answer, confidence score, abstention reason, and supporting metrics.
- **Evaluation report**: Run configuration and aggregate quality/coverage metrics.

## Success Criteria

### Measurable Outcomes

- **SC-001**: All candidate-ranking and confidence-gate tests pass without reducing coverage of existing branches.
- **SC-002**: A bounded evaluation run produces all five required metrics when answer labels are available.
- **SC-003**: Candidates with verifier confidence below 0.10 are never accepted as answers.
- **SC-004**: A real Kaggle run records model-load status, GPU configuration, runtime failures, and quality metrics rather than claiming unmeasured improvement.

## Assumptions

- The existing Qwen/vLLM inference path remains the runtime under evaluation.
- The official evaluator accepts the existing answer schema and distinguishes semantic answer types.
- Thresholds remain configuration values and are tuned from measured runs rather than hard-coded to a single sample.
- Full GPU evaluation occurs in Kaggle; local tests validate logic and packaging only.
