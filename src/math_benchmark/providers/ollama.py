import base64
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from math_benchmark.providers.base import (
    ModelConfig,
    ModelIdentity,
    PromptConfig,
    ProviderProtocolError,
    ProviderResponse,
    ProviderTimeoutError,
    ProviderUnavailableError,
    render_prompt,
)
from math_benchmark.schemas import Problem


class OllamaProvider:
    def __init__(
        self,
        base_url: str,
        client: httpx.Client,
        data_root: Path = Path("data"),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client
        self.data_root = data_root
        self.clock = clock or (lambda: datetime.now(UTC))

    def _post(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        try:
            response = self.client.post(f"{self.base_url}{path}", json=payload, timeout=timeout)
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(f"Ollama request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"Ollama is unavailable: {exc}") from exc

        if response.status_code >= 500:
            raise ProviderUnavailableError(
                f"Ollama returned HTTP {response.status_code}: {response.text}"
            )
        if response.status_code >= 400:
            raise ProviderProtocolError(
                f"Ollama rejected the request with HTTP {response.status_code}: {response.text}"
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderProtocolError("Ollama returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ProviderProtocolError("Ollama response must be a JSON object")
        return data

    def resolve_model(self, model: ModelConfig) -> ModelIdentity:
        data = self._post("/api/show", {"model": model.tag}, model.timeout_seconds)
        digest = data.get("digest")
        if not isinstance(digest, str) or not digest.strip():
            raise ProviderProtocolError("Ollama model metadata is missing a nonempty digest")
        return ModelIdentity(tag=model.tag, digest=digest)

    def _image_path(self, relative_or_absolute: str) -> Path:
        path = Path(relative_or_absolute)
        return path if path.is_absolute() else self.data_root / path

    def generate(
        self, problem: Problem, prompt: PromptConfig, model: ModelConfig
    ) -> ProviderResponse:
        rendered = render_prompt(problem, prompt)
        message: dict[str, Any] = {"role": "user", "content": rendered}
        image_paths: tuple[str, ...] = ()
        if problem.figure_path is not None:
            image_path = self._image_path(problem.figure_path)
            try:
                encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            except OSError as exc:
                raise ProviderProtocolError(f"cannot read image {image_path}: {exc}") from exc
            message["images"] = [encoded]
            image_paths = (problem.figure_path,)

        payload = {
            "model": model.tag,
            "messages": [message],
            "stream": False,
            "options": model.generation_options(),
        }
        started_at = self.clock()
        data = self._post("/api/chat", payload, model.timeout_seconds)
        completed_at = self.clock()
        try:
            raw_text = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise ProviderProtocolError("Ollama response is missing message content") from exc
        if not isinstance(raw_text, str):
            raise ProviderProtocolError("Ollama message content must be a string")
        return ProviderResponse(
            raw_text=raw_text,
            provider=model.provider,
            model_tag=model.tag,
            started_at=started_at,
            completed_at=completed_at,
            rendered_prompt=rendered,
            image_paths=image_paths,
            generation_parameters=model.generation_options(),
        )
