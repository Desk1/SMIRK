#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path>"
    echo "Example: $0 checkpoints/model_final.pt"
    exit 1
fi

REMOTE_PATH="$1"

TEMP="SMILE-baseline/blackbox_attack/ours-surrogate_model/inception_resnetv1_vggface2-celeba_partial256-NGOpt-2500-2500-SMILE-400-0.2-inception_resnetv1_casia-vggface2->CASIA-3/LOGS"

scp -P 1024 "u03kj22@127.0.0.1:/home/u03kj22/sharedscratch/SMIRK/${TEMP}/${REMOTE_PATH}" "./${TEMP}/${REMOTE_PATH}"