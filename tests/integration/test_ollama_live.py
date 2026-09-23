import os
from pathlib import Path

import httpx
import pytest

from math_benchmark.dataset import load_problems
from math_benchmark.providers.base import ModelConfig
from math_benchmark.providers.ollama import OllamaProvider
from tests.unit.providers.test_ollama import prompt_config


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("OLLAMA_INTEGRATION") != "1", reason="live Ollama test is opt-in")
def test_live_ollama_on_one_problem() -> None:
    root = Path(__file__).parents[2]
    problem = next(
        problem
        for problem in load_problems(root / "data" / "problems.jsonl")
        if problem.problem_id == "ALG-F-001"
    )
    model = os.getenv("OLLAMA_TEST_MODEL", "gemma3:4b")
    timeout = 300.0
    with httpx.Client(timeout=timeout) as client:
        provider = OllamaProvider(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), client)
        identity = provider.resolve_model(ModelConfig(tag=model, timeout_seconds=timeout))
        response = provider.generate(problem, prompt_config(), ModelConfig(tag=identity.tag))
    assert response.raw_text.strip()
