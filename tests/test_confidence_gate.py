from src.modules.confidence_gate import ConfidenceGate
from src.modules.question_understanding import QuestionInfo


def test_repeated_matching_answers_are_accepted():
    gate = ConfidenceGate(min_confidence=0.3)

    result = gate.evaluate_runs(
        [
            {"type": "number", "answer": 100},
            {"type": "number", "answer": 100},
            {"type": "number", "answer": 100},
            {"type": "number", "answer": 100},
            {"type": "number", "answer": 100},
        ],
        question_info=None,
    )

    assert result == {"type": "number", "answer": 100}


def test_different_answer_types_select_modal_answer_with_relaxed_penalty():
    gate = ConfidenceGate(min_confidence=0.3)

    result = gate.evaluate_runs(
        [
            {"type": "number", "answer": 100},
            {"type": "string", "answer": "100"},
            {"type": "number", "answer": 100},
            {"type": "string", "answer": "100"},
            {"type": "abstain"},
        ],
        failed_runs=1,
        question_info=None,
    )

    assert result == {"type": "number", "answer": 100}


def test_numeric_dispersion_does_not_add_penalty():
    gate = ConfidenceGate(min_confidence=0.5)

    result = gate.evaluate_runs(
        [
            {"type": "number", "answer": 100},
            {"type": "number", "answer": 1000},
            {"type": "number", "answer": 2000},
            {"type": "number", "answer": 3000},
            {"type": "number", "answer": 4000},
        ],
        question_info=None,
    )

    assert result["type"] == "abstain"
    assert "dispersion_penalty" not in result["reason"]


def test_all_distinct_answers_are_abstained_without_consensus():
    gate = ConfidenceGate(min_confidence=0.3)

    result = gate.evaluate_runs(
        [{"type": "number", "answer": value} for value in range(100, 105)],
        question_info=None,
    )

    assert result["type"] == "abstain"
    assert "no_consensus" in result["reason"]


def test_prompt_verifier_can_reject_ambiguous_answers():
    gate = ConfidenceGate(min_confidence=0.3)

    def verifier(prompt):
        assert "All candidate answers" in prompt
        return '{"is_consistent": false, "confidence": 0.1, "reason": "no consensus"}'

    result = gate.evaluate_runs(
        [
            {"type": "number", "answer": 100},
            {"type": "number", "answer": 101},
            {"type": "number", "answer": 102},
            {"type": "number", "answer": 103},
            {"type": "number", "answer": 104},
        ],
        prompt_verifier_fn=verifier,
        verifier_context={"question": "test"},
    )

    assert result["type"] == "abstain"
    assert "prompt_verifier_inconsistent" in result["reason"]


def test_invalid_prompt_verifier_response_keeps_heuristic_result():
    gate = ConfidenceGate(min_confidence=0.3)

    result = gate.evaluate_runs(
        [{"type": "number", "answer": 100}] * 5,
        prompt_verifier_fn=lambda prompt: "not json",
    )

    assert result == {"type": "number", "answer": 100}


def test_candidate_verifier_ranks_semantically_better_candidate():
    gate = ConfidenceGate(min_confidence=0.3)

    def verifier(prompt):
        if '"answer": 200' in prompt:
            return '{"is_consistent": true, "confidence": 0.9, "reason": "matches rules"}'
        return '{"is_consistent": false, "confidence": 0.2, "reason": "wrong candidate"}'

    result = gate.evaluate_runs(
        [
            {"type": "number", "answer": 100},
            {"type": "number", "answer": 200},
        ],
        candidate_records=[
            {
                "answer": {"type": "number", "answer": 100},
                "execution_success": True,
                "retries": 0,
            },
            {
                "answer": {"type": "number", "answer": 200},
                "execution_success": True,
                "retries": 0,
            },
        ],
        question_info=QuestionInfo(expected_type="number", confidence=0.9),
        prompt_verifier_fn=verifier,
        verifier_context={"question": "test", "expected_type": "number"},
    )

    assert result == {"type": "number", "answer": 200}


def test_candidate_verifier_confidence_is_the_dominant_signal():
    gate = ConfidenceGate(min_confidence=0.3)

    def verifier(prompt):
        if '"answer": 200' in prompt:
            return '{"is_consistent": true, "confidence": 0.9, "reason": "correct"}'
        return '{"is_consistent": true, "confidence": 0.1, "reason": "uncertain"}'

    result = gate.evaluate_runs(
        [
            {"type": "number", "answer": 100},
            {"type": "number", "answer": 100},
            {"type": "number", "answer": 100},
            {"type": "number", "answer": 200},
        ],
        candidate_records=[
            {
                "answer": {"type": "number", "answer": 100},
                "execution_success": True,
                "retries": 0,
            },
            {
                "answer": {"type": "number", "answer": 100},
                "execution_success": True,
                "retries": 0,
            },
            {
                "answer": {"type": "number", "answer": 100},
                "execution_success": True,
                "retries": 0,
            },
            {
                "answer": {"type": "number", "answer": 200},
                "execution_success": True,
                "retries": 0,
            },
        ],
        question_info=QuestionInfo(expected_type="number", confidence=0.9),
        prompt_verifier_fn=verifier,
        verifier_context={"question": "test", "expected_type": "number"},
    )

    assert result == {"type": "number", "answer": 200}


def test_question_completeness_rejects_missing_context():
    gate = ConfidenceGate(min_confidence=0.3)

    result = gate.evaluate_question_completeness(
        lambda prompt: (
            '{"question_status":"unanswerable","confidence":0.9,'
            '"missing_context":["time_range"],"assumptions":[],'
            '"reason":"time range changes the result"}'
        ),
        question="Doanh thu món nào nhiều nhất?",
        as_of_date="2026-11-03",
        question_info=QuestionInfo(
            original="Doanh thu món nào nhiều nhất?",
            expected_type="string",
            confidence=0.9,
        ),
        schema="orders(product, amount)",
        rulebook="No default time range.",
    )

    assert result["question_status"] == "unanswerable"
    assert result["missing_context"] == ["time_range"]


def test_question_completeness_invalid_response_uses_fallback():
    gate = ConfidenceGate(min_confidence=0.3)

    result = gate.evaluate_question_completeness(
        lambda prompt: "not json",
        question="Tổng doanh thu?",
        as_of_date="2026-11-03",
        question_info=QuestionInfo(expected_type="number", confidence=0.8),
        schema="orders(amount)",
        rulebook="",
    )

    assert result["question_status"] == "answerable"
    assert result["confidence"] == 0.8
