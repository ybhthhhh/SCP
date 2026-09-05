#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 CONFIG_YAML OUTPUT_DIR" >&2
  exit 2
fi

CONFIG=$1
OUTPUT_DIR=$2
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

latest=$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)
if [ -n "${latest}" ]; then
  echo "Resuming from ${latest}"
  FORCE_TORCHRUN=1 NPROC_PER_NODE="${NPROC_PER_NODE}" \
    llamafactory-cli train "${CONFIG}" resume_from_checkpoint="${latest}"
else
  FORCE_TORCHRUN=1 NPROC_PER_NODE="${NPROC_PER_NODE}" \
    llamafactory-cli train "${CONFIG}"
fi
