import json

from math_benchmark.enums import AnswerType, ValidatorKind
from math_benchmark.schemas import GroundTruth, ValidationSpec
from math_benchmark.validators.models import ExtractedAnswer
from math_benchmark.validators.registry import validate_answer
from math_benchmark.validators.structured import structured_text_equivalent
from tests.unit.test_dataset import make_problem


def test_structured_json_ignores_object_key_order() -> None:
    expected = '{"valid": true, "conclusion": "Q", "assignments": {"P": true}}'
    actual = '{"assignments": {"P": true}, "conclusion": "Q", "valid": true}'
    assert structured_text_equivalent(actual, expected).is_correct is True


def test_structured_json_values_must_match() -> None:
    expected = '{"valid": true, "conclusion": "Q"}'
    actual = '{"conclusion": "Q", "valid": false}'
    assert structured_text_equivalent(actual, expected).is_correct is False


def test_structured_json_can_ignore_extra_keys() -> None:
    expected = '{"valid": true, "conclusion": "Q"}'
    actual = '{"valid": true, "conclusion": "Q", "explanation": "modus ponens"}'
    assert structured_text_equivalent(actual, expected).is_correct is True


def test_malformed_structured_answer_requires_review() -> None:
    canonical = {"valid": True, "conclusion": "Q", "assignments": {"P": True}}
    problem = make_problem().model_copy(
        update={
            "answer_type": AnswerType.STRUCTURED_TEXT,
            "ground_truth": GroundTruth(display=json.dumps(canonical), canonical=json.dumps(canonical)),
            "validation": ValidationSpec(validator=ValidatorKind.STRUCTURED),
        }
    )
    result = validate_answer(problem, ExtractedAnswer("valid: yes", False, None))
    assert result.is_correct is None
    assert result.needs_review is True
