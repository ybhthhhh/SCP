#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RAW_DIR=${RAW_DIR:-${PROJECT_ROOT}/data/raw}
mkdir -p "${RAW_DIR}"

wget -c -O "${RAW_DIR}/light-r1-stage2-3k.json" \
  "https://hf-mirror.com/datasets/qihoo360/Light-R1-SFTData/resolve/main/stage2-3k.json"
wget -c -O "${RAW_DIR}/dapo-math-17k.parquet" \
  "https://hf-mirror.com/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet"
wget -c -O "${RAW_DIR}/aime-2024.parquet" \
  "https://hf-mirror.com/datasets/BytedTsinghua-SIA/AIME-2024/resolve/main/data/aime-2024.parquet"

python3 - "${RAW_DIR}/light-r1-stage2-3k.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    rows = json.load(handle)
print(f"Light-R1 rows: {len(rows)} (current upstream has 3,533; paper reports 3,335 after preprocessing)")
PY
