#!/usr/bin/env bash
set -euo pipefail

download_and_unzip() {
    local url="$1"
    local gz_file="$2"
    local txt_file="$3"

    if [[ -f "$txt_file" ]]; then
        echo "$txt_file already exists, skipping."
        return
    fi

    if [[ -f "$gz_file" ]]; then
        echo "$gz_file already exists, unzipping to $txt_file..."
    else
        echo "Downloading $gz_file..."
        wget -O "$gz_file" "$url"
    fi

    gunzip -f "$gz_file"
}

download_and_unzip \
  "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz" \
  "owt_train.txt.gz" \
  "owt_train.txt"

download_and_unzip \
  "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz" \
  "owt_valid.txt.gz" \
  "owt_valid.txt"
