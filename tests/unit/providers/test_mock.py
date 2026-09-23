from datetime import UTC, datetime

from math_benchmark.providers.base import ModelConfig, PromptConfig
from math_benchmark.providers.mock import MockProvider
from tests.unit.test_dataset import make_problem


def test_mock_provider_uses_problem_id_lookup() -> None:
    instants = iter(
        [
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
            datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC),
            datetime(2026, 1, 1, 0, 0, 3, tzinfo=UTC),
        ]
    )
    provider = MockProvider(
        {"ALG-F-001": "Final answer: 6", "ALG-F-002": "Final answer: 7"},
        clock=lambda: next(instants),
    )
    prompt = PromptConfig(version="1.0.0", base_instruction="Solve.\nFinal answer: <answer>")
    model = ModelConfig(tag="mock:1", provider="mock")

    second = make_problem("ALG-F-002")
    first = make_problem("ALG-F-001")
    assert provider.generate(second, prompt, model).raw_text == "Final answer: 7"
    response = provider.generate(first, prompt, model)
    assert response.raw_text == "Final answer: 6"
    assert response.started_at == datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC)
    assert response.completed_at == datetime(2026, 1, 1, 0, 0, 3, tzinfo=UTC)


def test_mock_resolves_stable_identity() -> None:
    provider = MockProvider({})
    identity = provider.resolve_model(ModelConfig(tag="mock:1", provider="mock"))
    assert identity.tag == "mock:1"
    assert identity.digest == "sha256:mock"

