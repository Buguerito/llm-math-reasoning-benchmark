from datetime import UTC, datetime
from pathlib import Path

import pytest

from math_benchmark.enums import RunStatus
from math_benchmark.schemas import RunAttempt
from math_benchmark.storage import JsonlAttemptStore, StorageError


def attempt(run_id: str, raw_response: str | None = None) -> RunAttempt:
    return RunAttempt(
        run_id=run_id,
        problem_id="ALG-F-001",
        benchmark_version="1.0.0",
        prompt_version="1.0.0",
        provider="mock",
        model_name="mock:1",
        model_digest="sha256:mock",
        rendered_prompt="Solve.",
        raw_response=raw_response,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        completed_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        latency_seconds=1.0,
        status=RunStatus.SUCCESS,
    )


def test_append_never_rewrites_existing_attempt(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    store = JsonlAttemptStore(path)
    store.append(attempt(run_id="run-1", raw_response="first"))
    before = path.read_bytes()
    store.append(attempt(run_id="run-2", raw_response="second"))
    assert path.read_bytes().startswith(before)
    assert [item.run_id for item in store.load_all()] == ["run-1", "run-2"]


def test_duplicate_run_id_is_rejected(tmp_path: Path) -> None:
    store = JsonlAttemptStore(tmp_path / "attempts.jsonl")
    store.append(attempt(run_id="same"))
    with pytest.raises(StorageError, match="duplicate run_id"):
        store.append(attempt(run_id="same"))

