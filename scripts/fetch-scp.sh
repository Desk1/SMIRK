#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path>"
    echo "Example: $0 checkpoints/model_final.pt"
    exit 1
fi

REMOTE_PATH="$1"

scp -P 1024 "u03kj22@127.0.0.1:/home/u03kj22/sharedscratch/SMIRK/${REMOTE_PATH}" ./