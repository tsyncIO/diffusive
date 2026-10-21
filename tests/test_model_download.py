"""Test suite for validating model links and download_models.sh functionality."""

import subprocess
import urllib.request
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_LINKS_FILE = PROJECT_ROOT / "scripts" / "model_links.txt"
DOWNLOAD_SCRIPT = PROJECT_ROOT / "scripts" / "download_models.sh"


def get_model_entries():
    """Parse scripts/model_links.txt into (url, target_folder) pairs."""
    entries = []
    assert MODEL_LINKS_FILE.exists(), f"Links file missing: {MODEL_LINKS_FILE}"
    for line in MODEL_LINKS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        url = parts[0]
        target_folder = parts[1] if len(parts) > 1 else "models"
        entries.append((url, target_folder))
    return entries


def test_model_links_file_syntax():
    """Verify that model_links.txt contains valid URLs and target folders."""
    entries = get_model_entries()
    assert len(entries) > 0, "No entries found in model_links.txt"
    for url, folder in entries:
        assert url.startswith("http://") or url.startswith("https://"), f"Invalid URL scheme: {url}"
        assert folder.startswith("models"), f"Target folder should be inside models/, got: {folder}"


@pytest.mark.network
def test_sample_model_url_accessibility():
    """Perform a lightweight HTTP HEAD/Range check on sample URLs to verify remote availability."""
    entries = get_model_entries()
    # Check first 2 URLs to avoid hitting rate limits while verifying host availability
    for url, _ in entries[:2]:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                assert resp.status in (200, 301, 302, 303, 307, 308)
        except urllib.error.HTTPError as e:
            # Some CDNs return 403 or 405 on HEAD but accept GET
            assert e.code in (200, 301, 302, 403, 405)


def test_download_models_script_execution(tmp_path):
    """Test that download_models.sh cleanly handles missing links, custom files, and idempotent downloads."""
    test_links = tmp_path / "test_links.txt"
    # Use a small raw GitHub text file as a mock model file
    mock_url = "https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/.gitignore"
    test_links.write_text(f"{mock_url} | models/test_dest\n")

    # Run downloader targeting the test file
    result = subprocess.run(
        ["bash", str(DOWNLOAD_SCRIPT), str(test_links)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    downloaded_file = PROJECT_ROOT / "models" / "test_dest" / ".gitignore"
    try:
        assert downloaded_file.exists(), "Target file was not downloaded"
        assert downloaded_file.stat().st_size > 0, "Downloaded file is empty"

        # Verify idempotency (should skip already downloaded file)
        result_idempotent = subprocess.run(
            ["bash", str(DOWNLOAD_SCRIPT), str(test_links)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result_idempotent.returncode == 0
        assert "Already exists" in result_idempotent.stdout
    finally:
        # Clean up mock test directory
        test_dir = PROJECT_ROOT / "models" / "test_dest"
        if test_dir.exists():
            for f in test_dir.iterdir():
                f.unlink()
            test_dir.rmdir()
