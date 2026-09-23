from collections.abc import Callable, Mapping
from datetime import UTC, datetime

from math_benchmark.providers.base import (
    ModelConfig,
    ModelIdentity,
    PromptConfig,
    ProviderProtocolError,
    ProviderResponse,
    render_prompt,
)
from math_benchmark.schemas import Problem


class MockProvider:
    def __init__(
        self,
        responses: Mapping[str, str],
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._responses = dict(responses)
        self._clock = clock or (lambda: datetime.now(UTC))

    def resolve_model(self, model: ModelConfig) -> ModelIdentity:
        return ModelIdentity(tag=model.tag, digest="sha256:mock")

    def generate(
        self, problem: Problem, prompt: PromptConfig, model: ModelConfig
    ) -> ProviderResponse:
        try:
            raw_text = self._responses[problem.problem_id]
        except KeyError as exc:
            raise ProviderProtocolError(
                f"mock response missing for problem {problem.problem_id}"
            ) from exc
        started_at = self._clock()
        completed_at = self._clock()
        return ProviderResponse(
            raw_text=raw_text,
            provider=model.provider,
            model_tag=model.tag,
            started_at=started_at,
            completed_at=completed_at,
            rendered_prompt=render_prompt(problem, prompt),
            image_paths=tuple(filter(None, [problem.figure_path])),
            generation_parameters=model.generation_options(),
        )

