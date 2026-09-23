from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedAnswer:
    text: str | None
    needs_review: bool
    reason: str | None


@dataclass(frozen=True)
class ValidationResult:
    is_correct: bool | None
    needs_review: bool
    normalized_actual: str | None
    reason: str

