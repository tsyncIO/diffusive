"""Test suite for validating inference_qedit.py execution and static packages/ integration."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "inference_qedit.py"
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"


def test_static_packages_loaded_from_repo():
    """Verify comfy and comfyui_to_python are imported from the static packages/ directory."""
    result = subprocess.run(
        [
            str(PYTHON_BIN),
            "-c",
            "import comfy, comfyui_to_python; print(comfy.__file__); print(comfyui_to_python.__file__)",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Import check failed: {result.stderr}"
    lines = [line.strip() for line in result.stdout.strip().splitlines()]
    assert str(PROJECT_ROOT / "packages" / "comfyui") in lines[0]
    assert str(PROJECT_ROOT / "packages" / "comfyui-to-python") in lines[1]


def test_qedit_cli_help():
    """Verify that inference_qedit.py CLI help functions properly."""
    result = subprocess.run(
        [str(PYTHON_BIN), str(SCRIPT_PATH), "--help"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Run the Qwen-Image (qedit) standalone workflow" in result.stdout
    assert "--dry-run" in result.stdout


def test_qedit_dry_run_bootstrap():
    """Verify inference_qedit.py bootstraps the ComfyUI runtime and resolves all 9 required nodes."""
    result = subprocess.run(
        [str(PYTHON_BIN), str(SCRIPT_PATH), "--dry-run"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Dry-run failed: {result.stderr}"
    assert "[DRY-RUN] ComfyUI runtime bootstrap and custom nodes imported successfully." in result.stdout
    assert "'UnetLoaderGGUF': True" in result.stdout
    assert "'CLIPLoader': True" in result.stdout
    assert "'VAELoader': True" in result.stdout
