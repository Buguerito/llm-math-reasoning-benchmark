import re

from math_benchmark.validators.models import ExtractedAnswer

FINAL_ANSWER_PATTERN = re.compile(r"^\s*Final answer:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)


def extract_final_answer(response: str) -> ExtractedAnswer:
    """Extract the required final-answer line and flag ambiguous responses."""
    normalized = response.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return ExtractedAnswer(None, True, "empty response")

    matches = [match.strip() for match in FINAL_ANSWER_PATTERN.findall(normalized)]
    if matches:
        selected = matches[-1] or None
        distinct = set(matches)
        if len(distinct) > 1:
            return ExtractedAnswer(selected, True, "conflicting final-answer markers")
        if selected is None:
            return ExtractedAnswer(None, True, "empty final-answer marker")
        return ExtractedAnswer(selected, False, None)

    last_line = next((line.strip() for line in reversed(normalized.splitlines()) if line.strip()), None)
    return ExtractedAnswer(last_line, True, "required marker missing")

