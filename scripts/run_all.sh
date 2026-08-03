#!/usr/bin/env bash
# Full main-text reproduction for one model: Tables 1-4 and Figures 3-4.
#
#   scripts/run_all.sh configs/models/llama2-13b.yaml [gpu_index]
#
# Tables 1-3 use configs/main.yaml; the ablation and the sweeps use the smaller
# validation-split sizes of configs/ablation.yaml (see that file for why).
set -euo pipefail

CONFIG="${1:?usage: run_all.sh <model-config.yaml> [gpu]}"
GPU="${2:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

KEY="$(python3 -c "
import sys; sys.path.insert(0,'.')
from coft.registry import load_config
print(load_config('$CONFIG')['model']['key'])
")"
LOG_DIR="logs/$KEY"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
# Keep the per-stage logs readable while a multi-hour run is in flight:
# Python block-buffers stdout when it is redirected to a file.
export PYTHONUNBUFFERED=1
DEV='{"":0}'
COMMON=(--config "$CONFIG" --device-map "$DEV" --no-progress)

stage () {
  local name="$1"; shift
  echo "[$(date +%H:%M:%S)] === $KEY :: $name ==="
  if ! "$@" > "$LOG_DIR/$name.log" 2>&1; then
    echo "[$(date +%H:%M:%S)] !!! $name FAILED (see $LOG_DIR/$name.log)"
    tail -25 "$LOG_DIR/$name.log"
    return 1
  fi
  tail -n 12 "$LOG_DIR/$name.log"
}

stage table1 python3 scripts/run_bias.py       "${COMMON[@]}" --override configs/main.yaml
stage table2 python3 scripts/run_utility.py    "${COMMON[@]}" --override configs/main.yaml
stage table3 python3 scripts/run_efficiency.py "${COMMON[@]}" --override configs/main.yaml
stage table4 python3 scripts/run_ablation.py   "${COMMON[@]}" --override configs/ablation.yaml
stage sweeps python3 scripts/run_sweep.py      "${COMMON[@]}" --override configs/ablation.yaml

echo "[$(date +%H:%M:%S)] === $KEY :: done ==="
