from math_benchmark.environment import collect_environment


def test_environment_capture_serializes_all_fixed_fields() -> None:
    outputs = {
        ("python", "--version"): "Python 3.12.1",
        ("platform",): "Windows-11-test",
        ("cpu",): "Test CPU",
        ("ram",): "17179869184",
        ("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"): "RTX 2060 SUPER, 555.1, 8192 MiB",
        ("ollama", "--version"): "ollama version 0.12.0",
        ("python", "-m", "pip", "freeze"): "httpx==0.28.1\npydantic==2.13.5",
    }

    snapshot = collect_environment(lambda command: outputs[tuple(command)])
    assert snapshot.model_dump(mode="json") == {
        "python_version": "Python 3.12.1",
        "platform": "Windows-11-test",
        "cpu": "Test CPU",
        "total_ram_bytes": 17179869184,
        "gpu": "RTX 2060 SUPER, 555.1, 8192 MiB",
        "ollama_version": "ollama version 0.12.0",
        "packages": ["httpx==0.28.1", "pydantic==2.13.5"],
    }


def test_missing_nvidia_smi_yields_no_gpu() -> None:
    def runner(command) -> str:
        if command[0] == "nvidia-smi":
            raise FileNotFoundError("missing")
        defaults = {
            "python": "Python 3.12.1",
            "platform": "test-os",
            "cpu": "test-cpu",
            "ram": "1024",
            "ollama": "ollama 0.12",
        }
        return defaults[command[0]]

    assert collect_environment(runner).gpu is None

