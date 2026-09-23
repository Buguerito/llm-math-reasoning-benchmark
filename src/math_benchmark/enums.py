from enum import IntEnum, StrEnum


class Category(StrEnum):
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    PROBABILITY = "probability"
    LOGIC = "logic"
    GRAPH_INTERPRETATION = "graph_interpretation"
    WORD_PROBLEMS = "word_problems"


class Difficulty(StrEnum):
    FOUNDATIONAL = "foundational"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class AnswerType(StrEnum):
    EXACT = "exact"
    NUMERIC = "numeric"
    RATIONAL = "rational"
    SYMBOLIC = "symbolic"
    SET_INTERVAL = "set_interval"
    STRUCTURED_TEXT = "structured_text"


class ValidatorKind(StrEnum):
    EXACT = "exact"
    NUMERIC = "numeric"
    SYMBOLIC = "symbolic"
    SET_INTERVAL = "set_interval"
    STRUCTURED = "structured"


class SourceType(StrEnum):
    AUTHORED = "authored"
    ADAPTED = "adapted"
    PUBLIC = "public"


class ErrorCategory(StrEnum):
    CORRECT = "correct"
    ARITHMETIC_ALGEBRAIC = "arithmetic_or_algebraic_error"
    LOGICAL_INFERENCE = "logical_inference_error"
    INCORRECT_ASSUMPTION = "incorrect_assumption"
    MISINTERPRETATION = "problem_misinterpretation"
    INCOMPLETE_REASONING = "incomplete_reasoning"
    UNSUPPORTED_CLAIM = "unsupported_claim_or_hallucination"
    INSTRUCTION_FAILURE = "instruction_following_failure"


class RunStatus(StrEnum):
    SUCCESS = "success"
    PROVIDER_ERROR = "provider_error"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"
    CANCELLED = "cancelled"


class ExperimentKind(StrEnum):
    PRIMARY = "primary"
    STABILITY = "stability"


class CorrectnessScore(IntEnum):
    INCORRECT = 0
    MINIMAL = 1
    PARTIAL = 2
    MOSTLY_CORRECT = 3
    FULLY_CORRECT = 4


class ReasoningQualityScore(IntEnum):
    NONE = 0
    POOR = 1
    MIXED = 2
    SOUND = 3
    EXCELLENT = 4


class InstructionFollowingScore(IntEnum):
    FAILED = 0
    PARTIAL = 1
    COMPLETE = 2

