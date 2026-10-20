#!/usr/bin/env bash

# Enhanced Model Downloader - Downloads to project root models/ folders
# Format in model_links.txt: URL | TARGET_FOLDER

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LINKS_FILE="${1:-$SCRIPT_DIR/model_links.txt}"
BASE_DIR="$PROJECT_ROOT"

if [[ ! -f "$LINKS_FILE" ]]; then
    echo "❌ Error: Model links file not found: $LINKS_FILE"
    exit 1
fi

echo "Starting download process..."
echo "Project root directory: $BASE_DIR"
echo "Links file: $LINKS_FILE"

while IFS='|' read -r url folder || [[ -n "$url" ]]; do
    # Trim whitespace
    url=$(echo "$url" | xargs)
    folder=$(echo "$folder" | xargs)

    # Skip empty lines or comments
    [[ -z "$url" || "$url" =~ ^# ]] && continue

    filename=$(basename "${url%%\?*}")

    # Determine target directory
    if [[ -n "$folder" ]]; then
        if [[ "$folder" == /* ]]; then
            target_dir="$folder"
        else
            target_dir="$BASE_DIR/$folder"
        fi
    else
        target_dir="$BASE_DIR/models"
    fi

    mkdir -p "$target_dir"
    target_file="$target_dir/$filename"

    if [[ -f "$target_file" ]]; then
        echo "✓ Already exists: $filename in $target_dir (skipping)"
        continue
    fi

    echo "Downloading: $filename to $target_dir"

    if command -v wget &> /dev/null; then
        wget -q --show-progress -c "$url" -O "$target_file" || rm -f "$target_file"
    else
        curl -L "$url" -o "$target_file" || rm -f "$target_file"
    fi

    if [[ -f "$target_file" && -s "$target_file" ]]; then
        echo "✓ Downloaded: $filename"
    else
        rm -f "$target_file"
        echo "✗ Failed: $filename"
    fi

done < "$LINKS_FILE"

echo "All downloads completed!"
