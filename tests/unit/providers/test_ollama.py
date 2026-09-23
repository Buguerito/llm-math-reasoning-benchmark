import base64
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from math_benchmark.providers.base import (
    ModelConfig,
    ModelIdentity,
    PromptConfig,
    ProviderProtocolError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from math_benchmark.providers.ollama import OllamaProvider
from tests.unit.test_dataset import make_problem

ROOT = Path(__file__).parents[3]


class RecordingTransport(httpx.BaseTransport):
    def __init__(self, handler) -> None:
        self.handler = handler
        self.last_json: dict[str, Any] | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.content:
            self.last_json = json.loads(request.content)
        return self.handler(request)


def prompt_config() -> PromptConfig:
    return PromptConfig(
        version="1.0.0",
        base_instruction=(
            "Provide a concise solution that explains the key steps.\n"
            "End your response with: Final answer: <answer>"
        ),
    )


def model_config(tag: str) -> ModelConfig:
    return ModelConfig(
        tag=tag,
        provider="ollama",
        temperature=0,
        seed=42,
        num_ctx=4096,
        num_predict=2048,
        timeout_seconds=300,
    )


def test_ollama_text_payload() -> None:
    transport = RecordingTransport(
        lambda request: httpx.Response(200, json={"message": {"content": "Final answer: 6"}})
    )
    provider = OllamaProvider("http://localhost:11434", httpx.Client(transport=transport))
    response = provider.generate(make_problem(), prompt_config(), model_config("gemma3:4b"))
    assert response.raw_text == "Final answer: 6"
    assert transport.last_json is not None
    assert transport.last_json["model"] == "gemma3:4b"
    assert transport.last_json["stream"] is False
    assert transport.last_json["options"] == {
        "temperature": 0,
        "num_ctx": 4096,
        "num_predict": 2048,
        "seed": 42,
    }
    assert "images" not in transport.last_json["messages"][0]


def test_ollama_image_payload_contains_base64_png(tmp_path: Path) -> None:
    image_path = tmp_path / "plot.png"
    source_path = tmp_path / "plot.csv"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    source_path.write_text("x,y\n0,0\n", encoding="utf-8")
    graph_problem = make_problem().model_copy(
        update={"figure_path": str(image_path), "figure_source_path": str(source_path)}
    )
    transport = RecordingTransport(
        lambda request: httpx.Response(200, json={"message": {"content": "Final answer: {}"}})
    )
    provider = OllamaProvider("http://localhost:11434", httpx.Client(transport=transport))
    provider.generate(graph_problem, prompt_config(), model_config("qwen2.5vl:3b"))
    assert transport.last_json is not None
    images = transport.last_json["messages"][0]["images"]
    assert len(images) == 1
    decoded = base64.b64decode(images[0], validate=True)
    assert decoded.startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize(
    ("handler", "expected_error"),
    [
        (lambda request: httpx.Response(500, text="down"), ProviderUnavailableError),
        (lambda request: httpx.Response(200, content=b"not json"), ProviderProtocolError),
        (lambda request: httpx.Response(200, json={"message": {}}), ProviderProtocolError),
        (
            lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("slow", request=request)),
            ProviderTimeoutError,
        ),
    ],
)
def test_ollama_errors_are_typed(handler, expected_error: type[Exception]) -> None:
    provider = OllamaProvider(
        "http://localhost:11434", httpx.Client(transport=RecordingTransport(handler))
    )
    with pytest.raises(expected_error):
        provider.generate(make_problem(), prompt_config(), model_config("gemma3:4b"))


def test_resolve_model_records_ollama_digest() -> None:
    transport = RecordingTransport(
        lambda request: httpx.Response(200, json={"digest": "sha256:test-digest"})
    )
    provider = OllamaProvider("http://localhost:11434", httpx.Client(transport=transport))
    assert provider.resolve_model(model_config("gemma3:4b")) == ModelIdentity(
        tag="gemma3:4b", digest="sha256:test-digest"
    )


def test_versioned_configs_pin_models_and_profiles() -> None:
    prompts = yaml.safe_load((ROOT / "configs" / "prompts.yaml").read_text(encoding="utf-8"))
    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text(encoding="utf-8"))
    assert prompts["version"] == "1.0.0"
    assert prompts["base_instruction"].splitlines() == [
        "Provide a concise solution that explains the key steps.",
        "End your response with: Final answer: <answer>",
    ]
    assert [model["tag"] for model in models["models"]] == ["qwen2.5vl:3b", "gemma3:4b"]
    assert models["profiles"]["primary"]["seeds"] == [42]
    assert models["profiles"]["stability"]["seeds"] == [1103, 2207, 3301]

