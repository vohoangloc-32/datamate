# Tasks: Ranking and Abstention Calibration

## Phase 1 - Baseline

- [ ] T001 Confirm current confidence-gate tests cover verifier ranking, fallback consensus, low-confidence floors, and answer-type preservation.
- [ ] T002 Build a deterministic candidate matrix covering success, retry, type mismatch, implausible values, and disagreement.
- [ ] T003 Record the current local test count and notebook/archive synchronization status.

## Phase 2 - Evaluation Contract

- [ ] T004 Verify the evaluation report exposes accuracy, coverage, abstain rate, execution success, and candidate agreement.
- [ ] T005 Verify missing model, dataset, GPU, and CUDA prerequisites fail with actionable diagnostics.
- [ ] T006 Verify `unanswerable` is preserved and only confidence/execution failures produce `abstain`.

## Phase 3 - Integration

- [ ] T007 Run focused pytest and Python compilation checks for changed source modules.
- [ ] T008 Compile the Kaggle notebooks and verify configuration values, including sample size and candidate count.
- [ ] T009 Rebuild or validate `datamate_code.zip` against `src/` and `prompts/`.

## Phase 4 - Kaggle Validation

- [ ] T010 Run a small Kaggle smoke test with a bounded question limit.
- [ ] T011 Run the configured 500-question benchmark after the smoke test succeeds.
- [ ] T012 Compare metrics to the baseline and document whether threshold changes improve the required trade-off.

## Phase 5 - Closure

- [ ] T013 Run the final focused/full regression suite and `git diff --check`.
- [ ] T014 Update README/runtime notes with measured results and known limitations.
