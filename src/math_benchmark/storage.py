import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from math_benchmark.schemas import RunAttempt


class StorageError(RuntimeError):
    """Attempt or manifest storage is invalid."""


class JsonlAttemptStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load_all(self) -> list[RunAttempt]:
        if not self.path.exists():
            return []
        attempts: list[RunAttempt] = []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise StorageError(f"cannot read attempt store {self.path}: {exc}") from exc
        for line_number, line in enumerate(lines, 1):
            try:
                if line.strip():
                    attempts.append(RunAttempt.model_validate_json(line))
            except (ValidationError, ValueError) as exc:
                raise StorageError(f"invalid attempt store {self.path}:{line_number}: {exc}") from exc
        return attempts

    def append(self, attempt: RunAttempt) -> None:
        if any(existing.run_id == attempt.run_id for existing in self.load_all()):
            raise StorageError(f"duplicate run_id {attempt.run_id}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            attempt.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        with self.path.open("a", encoding="utf-8", newline="\n", buffering=1) as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def write_manifest_atomic(path: Path, manifest: BaseModel | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    value = manifest.model_dump(mode="json") if isinstance(manifest, BaseModel) else manifest
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)
