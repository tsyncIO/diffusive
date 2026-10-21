"""Test suite for validating custom node links and download_custom_nodes.sh."""

import subprocess
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NODE_LINKS_FILE = PROJECT_ROOT / "scripts" / "custom_nodes_links.txt"
DOWNLOAD_NODES_SCRIPT = PROJECT_ROOT / "scripts" / "download_custom_nodes.sh"


def get_node_links():
    """Extract all active git repo URLs from custom_nodes_links.txt."""
    links = []
    assert NODE_LINKS_FILE.exists(), f"Links file missing: {NODE_LINKS_FILE}"
    for line in NODE_LINKS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        links.append(line)
    return links


def test_custom_nodes_links_syntax():
    """Verify all custom node links are valid GitHub git repositories."""
    links = get_node_links()
    assert len(links) > 0, "No custom node links found in custom_nodes_links.txt"
    github_pattern = re.compile(r"^https://github\.com/[^/]+/[^/]+(\.git)?$")
    for link in links:
        assert github_pattern.match(link), f"Invalid GitHub link format: {link}"


def test_download_custom_nodes_with_empty_or_mock_list(tmp_path):
    """Verify download_custom_nodes.sh handles a single mock custom node repo cleanly."""
    mock_links_file = tmp_path / "mock_node_links.txt"
    # Use a small public repository as a test custom node
    mock_links_file.write_text("https://github.com/Comfy-Org/comfy-cli.git\n")

    # Run the script with the custom links file
    result = subprocess.run(
        ["bash", str(DOWNLOAD_NODES_SCRIPT), str(mock_links_file)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    target_repo = PROJECT_ROOT / "custom_nodes" / "comfy-cli"
    try:
        assert target_repo.exists(), "Custom node was not cloned"
        assert (target_repo / ".git").exists(), "Cloned repository missing .git"

        # Verify idempotency: running again recognizes it is already installed
        result_idempotent = subprocess.run(
            ["bash", str(DOWNLOAD_NODES_SCRIPT), str(mock_links_file)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        assert result_idempotent.returncode == 0
        assert "Already installed" in result_idempotent.stdout
    finally:
        # Clean up mock repo
        if target_repo.exists():
            subprocess.run(["rm", "-rf", str(target_repo)], check=True)
