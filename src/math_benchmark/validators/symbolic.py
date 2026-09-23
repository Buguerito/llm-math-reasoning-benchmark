import ast
import re
from typing import Any

import sympy  # type: ignore[import-untyped]

from math_benchmark.validators.bounded import run_bounded
from math_benchmark.validators.models import ValidationResult

SAFE_FUNCTIONS = {
    "sin": sympy.sin,
    "cos": sympy.cos,
    "exp": sympy.exp,
    "log": sympy.log,
    "sqrt": sympy.sqrt,
}
SAFE_CONSTANTS = {"pi": sympy.pi, "E": sympy.E}
SAFE_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
)


def _allowed_symbols(options: dict[str, Any]) -> set[str]:
    raw_symbols = options.get("symbols", [])
    if not isinstance(raw_symbols, list) or not all(
        isinstance(symbol, str) and symbol.isidentifier() and "_" not in symbol
        for symbol in raw_symbols
    ):
        raise ValueError("symbols option must be a list of safe identifiers")
    return set(raw_symbols)


def _check_source(source: str, symbols: set[str]) -> str | None:
    if len(source) > 512:
        return "symbolic input exceeds 512 characters"
    if re.search(r"[_\[\]'\"]", source):
        return "unsupported symbolic syntax"
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError:
        return "unsupported symbolic syntax"
    allowed_names = symbols | set(SAFE_FUNCTIONS) | set(SAFE_CONSTANTS)
    for node in ast.walk(tree):
        if not isinstance(node, SAFE_NODES):
            return "unsupported symbolic syntax"
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            return "unsupported symbolic syntax"
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTIONS
        ):
            return "unsupported symbolic syntax"
        if isinstance(node, ast.Constant) and not isinstance(node.value, int | float):
            return "unsupported symbolic syntax"
    return None


def _symbolic_worker(actual: str, expected: str, options: dict[str, Any]) -> bool:
    symbols = _allowed_symbols(options)
    local_dict: dict[str, object] = {
        **SAFE_FUNCTIONS,
        **SAFE_CONSTANTS,
        **{name: sympy.Symbol(name) for name in symbols},
    }
    actual_expr = sympy.sympify(actual, locals=local_dict, evaluate=True)
    expected_expr = sympy.sympify(expected, locals=local_dict, evaluate=True)
    return bool(sympy.simplify(actual_expr - expected_expr) == 0)


def symbolic_equivalent(
    actual: str,
    expected: str,
    options: dict[str, Any],
    timeout_seconds: float,
) -> ValidationResult:
    """Check symbolic equivalence using a strict syntax allowlist and child process."""
    try:
        symbols = _allowed_symbols(options)
    except ValueError as exc:
        return ValidationResult(None, True, actual.strip(), str(exc))
    for source in (actual, expected):
        error = _check_source(source, symbols)
        if error:
            return ValidationResult(None, True, actual.strip(), error)

    bounded = run_bounded(
        "math_benchmark.validators.symbolic._symbolic_worker",
        (actual, expected, options),
        timeout_seconds,
    )
    if bounded.timeout:
        return ValidationResult(None, True, actual.strip(), "symbolic validation timed out")
    if bounded.error is not None:
        return ValidationResult(None, True, actual.strip(), f"symbolic validation failed: {bounded.error}")
    return ValidationResult(bool(bounded.value), False, actual.strip(), "symbolic equivalence")
