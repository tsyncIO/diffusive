#!/usr/bin/env bash

# Custom Node Installer Script
# Reads GitHub links from a text file, clones missing custom nodes into <project_root>/custom_nodes,
# and installs their requirements into <project_root>/.venv using uv pip.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CN_DIR="$PROJECT_ROOT/custom_nodes"
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

echo "↳ Reading links from file..."
mapfile -t GITHUB_LINKS < <(grep -v '^#' "$LINKS_FILE" | grep -v '^$')

if [ ${#GITHUB_LINKS[@]} -eq 0 ]; then
    echo "❌ Error: No GitHub links found in $LINKS_FILE"
    exit 1
fi

echo "↳ Found ${#GITHUB_LINKS[@]} GitHub links in file"

EXISTING_NODES=0
NEW_NODES=0
NODES_TO_INSTALL=()

echo ""
echo "↳ Checking custom nodes…"

for link in "${GITHUB_LINKS[@]}"; do
    link_clean=$(echo "$link" | tr -d '[:space:]')
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
        ((EXISTING_NODES++))
    else
        echo "  📥 $node_name - Will be installed"
        NODES_TO_INSTALL+=("$link_clean|$node_name")
        ((NEW_NODES++))
    fi
done

echo ""
echo "↳ Check complete: $EXISTING_NODES existing, $NEW_NODES new"

if [ ${#NODES_TO_INSTALL[@]} -gt 0 ]; then
    echo ""
    echo "↳ Installing ${#NODES_TO_INSTALL[@]} new nodes…"

    for node_info in "${NODES_TO_INSTALL[@]}"; do
        IFS='|' read -r link node_name <<< "$node_info"
        target="$CN_DIR/$node_name"

        echo "  🔄 Cloning $node_name"
        echo "     From: $link"

        if git clone --depth 1 "$link" "$target" 2>/dev/null; then
            echo "  ✅ Successfully cloned $node_name"
        else
            echo "  ❌ Failed to clone $node_name"
        fi
        echo ""
    done

    echo "↳ Installation complete! Added ${#NODES_TO_INSTALL[@]} new nodes."
else
    echo ""
    echo "↳ All nodes are already installed. Nothing to do."
fi

echo ""
echo "↳ Recursively installing all requirements using uv..."
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
done < <(find "$CN_DIR" -maxdepth 2 -name "requirements.txt" -print0)

if [ -d "$CN_DIR/ComfyUI-to-Python-Extension" ]; then
    echo "  🔧 Installing ComfyUI-to-Python-Extension in editable mode..."
    "${UV_PIP_CMD[@]}" -e "$CN_DIR/ComfyUI-to-Python-Extension"
fi

echo ""
echo "📊 Summary:"
echo "  • Total nodes in file: ${#GITHUB_LINKS[@]}"
echo "  • Already installed: $EXISTING_NODES"
echo "  • Newly installed: $NEW_NODES"
echo "  • Custom nodes directory: $CN_DIR"

echo ""
echo "📁 Current custom_nodes directory contents:"
ls -la "$CN_DIR"
