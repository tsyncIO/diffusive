"""Test suite for validating inference_ifTalk.py execution without model inference."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "inference_ifTalk.py"
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"


def test_iftalk_cli_help():
    """Verify that inference_ifTalk.py CLI help functions properly."""
    result = subprocess.run(
        [str(PYTHON_BIN), str(SCRIPT_PATH), "--help"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Run the InfiniteTalk V2V workflow" in result.stdout
    assert "--dry-run" in result.stdout


def test_iftalk_dry_run_bootstrap():
    """Verify inference_ifTalk.py bootstraps the ComfyUI runtime and evaluates nodes without inference."""
    result = subprocess.run(
        [str(PYTHON_BIN), str(SCRIPT_PATH), "--dry-run"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Dry-run failed: {result.stderr}"
    assert "[DRY-RUN] ComfyUI runtime bootstrap and custom nodes imported successfully." in result.stdout
    assert "[DRY-RUN] Required node availability:" in result.stdout


def test_iftalk_programmatic_import_and_dry_run():
    """Directly test programmatic invocation of main(dry_run=True)."""
    # Import main from scripts.inference_ifTalk
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        from inference_ifTalk import main
        status = main(
            video="input/dummy.mp4",
            audio="input/dummy.wav",
            positive_prompt="test",
            negative_prompt="test",
            output_prefix="output/test",
            dry_run=True,
        )
        assert isinstance(status, dict)
        assert "WanVideoModelLoader" in status
        assert "WanVideoSampler" in status
    finally:
        if str(PROJECT_ROOT / "scripts") in sys.path:
            sys.path.remove(str(PROJECT_ROOT / "scripts"))
