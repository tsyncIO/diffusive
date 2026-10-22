#!/usr/bin/env bash

# Custom Node Installer Script
# Reads GitHub links (with optional `@ <commit_sha>`) from custom_nodes_links.txt,
# clones missing custom nodes into <project_root>/custom_nodes, links the vendored
# packages/comfyui-to-python extension, and installs requirements into .venv via uv.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CN_DIR="$PROJECT_ROOT/custom_nodes"
VENDORED_EXPORTER="$PROJECT_ROOT/packages/comfyui-to-python"
LINKS_FILE="${1:-$SCRIPT_DIR/custom_nodes_links.txt}"

echo "↳ Project root directory: $PROJECT_ROOT"
echo "↳ Custom nodes directory: $CN_DIR"
echo "↳ Links file: $LINKS_FILE"

if [ ! -f "$LINKS_FILE" ]; then
    echo "❌ Error: Links file not found: $LINKS_FILE"
    echo "Please create a text file with one GitHub URL per line"
    exit 1
fi

mkdir -p "$CN_DIR"

# Ensure vendored ComfyUI-to-Python-Extension is linked into custom_nodes/
if [ -d "$VENDORED_EXPORTER" ]; then
    ln -sfn "../packages/comfyui-to-python" "$CN_DIR/ComfyUI-to-Python-Extension"
    echo "↳ Linked static packages/comfyui-to-python -> custom_nodes/ComfyUI-to-Python-Extension"
fi

echo "↳ Reading links from file..."
mapfile -t GITHUB_LINKS < <(grep -v '^[[:space:]]*#' "$LINKS_FILE" | grep -v '^[[:space:]]*$')

EXISTING_NODES=0
NEW_NODES=0
NODES_TO_INSTALL=()

echo ""
echo "↳ Checking custom nodes…"

for entry in "${GITHUB_LINKS[@]}"; do
    # Parse optional `@ commit_sha`
    if [[ "$entry" == *"@"* ]]; then
        link_raw="${entry%%@*}"
        commit_raw="${entry#*@}"
    else
        link_raw="$entry"
        commit_raw=""
    fi

    link_clean=$(echo "$link_raw" | tr -d '[:space:]')
    commit_sha=$(echo "$commit_raw" | tr -d '[:space:]')

    if [ -z "$link_clean" ]; then
        continue
    fi

    if [[ $link_clean =~ github\.com/([^/]+)/([^/]+)(\.git)?$ ]]; then
        repo_name="${BASH_REMATCH[2]}"
        node_name="${repo_name%.git}"
    else
        echo "  ⚠️  Skipping invalid URL: $link_clean"
        continue
    fi

    target="$CN_DIR/$node_name"

    if [ -d "$target" ]; then
        echo "  ✅ $node_name - Already installed"
        if [ -n "$commit_sha" ] && [ -d "$target/.git" ]; then
            current_sha=$(git -C "$target" rev-parse HEAD 2>/dev/null)
            if [[ "$current_sha" != "$commit_sha"* ]]; then
                echo "     📌 Checking out pinned commit $commit_sha for $node_name..."
                git -C "$target" fetch --depth 1 origin "$commit_sha" 2>/dev/null || git -C "$target" fetch --unshallow 2>/dev/null
                git -C "$target" checkout "$commit_sha" 2>/dev/null
            fi
        fi
        ((EXISTING_NODES++))
    else
        echo "  📥 $node_name - Will be installed"
        NODES_TO_INSTALL+=("$link_clean|$node_name|$commit_sha")
        ((NEW_NODES++))
    fi
done

echo ""
echo "↳ Check complete: $EXISTING_NODES existing, $NEW_NODES new"

if [ ${#NODES_TO_INSTALL[@]} -gt 0 ]; then
    echo ""
    echo "↳ Installing ${#NODES_TO_INSTALL[@]} new nodes…"

    for node_info in "${NODES_TO_INSTALL[@]}"; do
        IFS='|' read -r link node_name commit_sha <<< "$node_info"
        target="$CN_DIR/$node_name"

        echo "  🔄 Cloning $node_name"
        echo "     From: $link"

        if git clone "$link" "$target" 2>/dev/null; then
            if [ -n "$commit_sha" ]; then
                echo "     📌 Pinning $node_name to commit $commit_sha"
                git -C "$target" checkout "$commit_sha" 2>/dev/null
            fi
            echo "  ✅ Successfully cloned $node_name"
        else
            echo "  ❌ Failed to clone $node_name"
        fi
        echo ""
    done

    echo "↳ Installation complete! Added ${#NODES_TO_INSTALL[@]} new nodes."
else
    echo ""
    echo "↳ All external custom nodes are already installed."
fi

echo ""
echo "↳ Recursively installing custom node requirements using uv..."
if [ -x "$PROJECT_ROOT/.venv/bin/uv" ]; then
    UV_PIP_CMD=("$PROJECT_ROOT/.venv/bin/uv" pip install --python "$PROJECT_ROOT/.venv/bin/python")
elif [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    UV_PIP_CMD=("$PROJECT_ROOT/.venv/bin/python" -m uv pip install)
else
    UV_PIP_CMD=(uv pip install)
fi

while IFS= read -r -d '' req_file; do
    echo "  🔧 Installing dependencies from: $req_file"
    "${UV_PIP_CMD[@]}" -r "$req_file"
done < <(find -L "$CN_DIR" -maxdepth 2 -name "requirements.txt" -print0)

echo ""
echo "📊 Summary:"
echo "  • External nodes in file: ${#GITHUB_LINKS[@]}"
echo "  • Already installed: $EXISTING_NODES"
echo "  • Newly installed: $NEW_NODES"
echo "  • Static vendored exporter: $VENDORED_EXPORTER"
echo "  • Custom nodes directory: $CN_DIR"

echo ""
echo "📁 Current custom_nodes directory contents:"
ls -la "$CN_DIR"
