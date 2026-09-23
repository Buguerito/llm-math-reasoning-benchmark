import ctypes
import os
import platform as platform_module
import subprocess
import sys
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict

CommandRunner = Callable[[Sequence[str]], str]


class EnvironmentSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    python_version: str
    platform: str
    cpu: str
    total_ram_bytes: int
    gpu: str | None
    ollama_version: str | None
    packages: list[str]


def _total_ram_bytes() -> int:
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return int(status.total_physical)
    page_size = os.sysconf("SC_PAGE_SIZE")
    pages = os.sysconf("SC_PHYS_PAGES")
    return int(page_size * pages)


def system_command_runner(command: Sequence[str]) -> str:
    """Run real commands plus portable synthetic system-info queries."""
    key = tuple(command)
    if key == ("platform",):
        return platform_module.platform()
    if key == ("cpu",):
        return platform_module.processor() or platform_module.machine()
    if key == ("ram",):
        return str(_total_ram_bytes())
    actual = [sys.executable, "--version"] if key == ("python", "--version") else list(command)
    if key[:3] == ("python", "-m", "pip"):
        actual[0] = sys.executable
    completed = subprocess.run(actual, check=True, capture_output=True, text=True)
    return (completed.stdout or completed.stderr).strip()


def _optional(runner: CommandRunner, command: Sequence[str]) -> str | None:
    try:
        value = runner(command).strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return value or None


def collect_environment(command_runner: CommandRunner) -> EnvironmentSnapshot:
    """Collect a serializable environment snapshot through an injectable runner."""
    python_version = command_runner(("python", "--version")).strip()
    platform_name = command_runner(("platform",)).strip()
    cpu = command_runner(("cpu",)).strip()
    total_ram = int(command_runner(("ram",)).strip())
    gpu = _optional(
        command_runner,
        ("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
    )
    ollama_version = _optional(command_runner, ("ollama", "--version"))
    packages_raw = command_runner(("python", "-m", "pip", "freeze"))
    packages = sorted(line.strip() for line in packages_raw.splitlines() if line.strip())
    return EnvironmentSnapshot(
        python_version=python_version,
        platform=platform_name,
        cpu=cpu,
        total_ram_bytes=total_ram,
        gpu=gpu,
        ollama_version=ollama_version,
        packages=packages,
    )
