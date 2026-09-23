from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from math_benchmark.schemas import Problem


class ProviderError(RuntimeError):
    """Base class for normalized provider failures."""


class ProviderUnavailableError(ProviderError):
    """The provider could not serve the request."""


class ProviderTimeoutError(ProviderError):
    """The provider request exceeded its deadline."""


class ProviderProtocolError(ProviderError):
    """The provider returned data that violates its contract."""


@dataclass(frozen=True)
class ModelConfig:
    tag: str
    provider: str = "ollama"
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 4096
    num_predict: int = 2048
    timeout_seconds: float = 300.0

    def generation_options(self) -> dict[str, int | float]:
        return {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class PromptConfig:
    version: str
    base_instruction: str


@dataclass(frozen=True)
class ModelIdentity:
    tag: str
    digest: str


@dataclass(frozen=True)
class ProviderResponse:
    raw_text: str
    provider: str
    model_tag: str
    started_at: datetime
    completed_at: datetime
    rendered_prompt: str
    image_paths: tuple[str, ...]
    generation_parameters: dict[str, int | float]

    @property
    def latency_seconds(self) -> float:
        return max(0.0, (self.completed_at - self.started_at).total_seconds())


def render_prompt(problem: Problem, prompt: PromptConfig) -> str:
    return f"{prompt.base_instruction}\n\n{problem.question}"


class Provider(Protocol):
    def resolve_model(self, model: ModelConfig) -> ModelIdentity: ...

    def generate(
        self, problem: Problem, prompt: PromptConfig, model: ModelConfig
    ) -> ProviderResponse: ...

