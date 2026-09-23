import importlib
import multiprocessing
from collections.abc import Sequence
from dataclasses import dataclass
from queue import Empty
from typing import Any


@dataclass(frozen=True)
class BoundedExecutionResult:
    value: object | None = None
    error: str | None = None
    timeout: bool = False

    @classmethod
    def timed_out(cls) -> "BoundedExecutionResult":
        return cls(timeout=True)


def _worker(callable_path: str, args: Sequence[object], queue: Any) -> None:
    try:
        module_name, callable_name = callable_path.rsplit(".", 1)
        function = getattr(importlib.import_module(module_name), callable_name)
        queue.put(("value", function(*args)))
    except Exception as exc:  # noqa: BLE001 - process boundary serializes all failures
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def run_bounded(
    callable_path: str, args: Sequence[object], timeout_seconds: float
) -> BoundedExecutionResult:
    """Execute an internal validator in a spawned process with a hard timeout."""
    if not callable_path.startswith("math_benchmark.validators."):
        return BoundedExecutionResult(error="callable path is outside validator package")
    if timeout_seconds <= 0:
        return BoundedExecutionResult(error="timeout must be positive")

    context = multiprocessing.get_context("spawn")
    queue = context.Queue(maxsize=1)
    process = context.Process(target=_worker, args=(callable_path, tuple(args), queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        queue.close()
        return BoundedExecutionResult.timed_out()

    try:
        kind, payload = queue.get(timeout=0.2)
    except Empty:
        return BoundedExecutionResult(error="validator worker returned no result")
    finally:
        queue.close()
    if kind == "error":
        return BoundedExecutionResult(error=str(payload))
    return BoundedExecutionResult(value=payload)
