from math_benchmark.validators.extraction import extract_final_answer
from math_benchmark.validators.models import ExtractedAnswer


def test_uses_single_required_marker() -> None:
    result = extract_final_answer("Work.\nFinal answer: 6")
    assert result == ExtractedAnswer(text="6", needs_review=False, reason=None)


def test_missing_marker_proposes_last_line_for_review() -> None:
    result = extract_final_answer("Work.\nTherefore x = 6")
    assert result.text == "Therefore x = 6"
    assert result.needs_review is True
    assert result.reason == "required marker missing"


def test_conflicting_repeated_markers_require_review() -> None:
    result = extract_final_answer("Final answer: 5\nCorrection.\nFinal answer: 6")
    assert result.text == "6"
    assert result.needs_review is True
    assert result.reason == "conflicting final-answer markers"


def test_repeated_equivalent_markers_are_accepted() -> None:
    result = extract_final_answer("Final answer: 6\nFinal answer:  6 ")
    assert result == ExtractedAnswer(text="6", needs_review=False, reason=None)


def test_empty_response_has_no_extractable_answer() -> None:
    assert extract_final_answer("   ").text is None

