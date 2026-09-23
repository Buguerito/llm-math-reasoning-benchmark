import re
from fractions import Fraction


def normalize_exact(value: str) -> str:
    """Apply only presentation-level normalization to an exact answer."""
    normalized = value.strip().replace("−", "-").replace("–", "-")
    normalized = re.sub(r"[.;,]+$", "", normalized).strip()
    if len(normalized) >= 2 and normalized.startswith("$") and normalized.endswith("$"):
        normalized = normalized[1:-1].strip()
    return " ".join(normalized.split())


def parse_fraction(value: str) -> Fraction:
    """Parse an integer, finite decimal, or simple rational safely."""
    normalized = normalize_exact(value).replace(" ", "")
    return Fraction(normalized)
