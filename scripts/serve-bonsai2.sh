#!/bin/bash
# Serve Ternary-Bonsai-2-27B with the fine-tuned DFlash 2 drafter on mlx-dspark.
#
#     scripts/serve-bonsai2.sh
#
# Builds the pinned environment (envs/dspark/uv.lock) in .venv, fetches the target and the
# drafter at pinned revisions, and serves an OpenAI-compatible API on 127.0.0.1:8088 with the
# settings the measurements used: 8-bit KV cache, drafter quantized to 4 bits at load, the
# target's published sampling. Needs macOS on Apple Silicon, uv, and about 13 GB of disk; long
# contexts want a 48 GB machine.
#
# Overrides: BONSAI2_VENV, BONSAI2_DRAFTER (a Hub repo or a local directory),
# BONSAI2_DRAFTER_REVISION, BONSAI2_HOST, BONSAI2_PORT.
set -euo pipefail
cd "$(dirname "$0")/.."

[ "$(uname -s)-$(uname -m)" = Darwin-arm64 ] || { echo "needs macOS on Apple Silicon" >&2; exit 1; }
command -v uv >/dev/null || { echo "needs uv: https://docs.astral.sh/uv/" >&2; exit 1; }

VENV=${BONSAI2_VENV:-$PWD/.venv}
TARGET=prism-ml/Ternary-Bonsai-2-27B-mlx-2bit
TARGET_REVISION=3f926b415992eaa2ae9dd7b573706494d6bbf787
DRAFTER=${BONSAI2_DRAFTER:-Schiltmans/Ternary-Bonsai-2-27B-DFlash2-ft5}
# The pinned revision belongs to the default drafter only. Another repository, such as the
# 4-bit variant, has its own commits, so it defaults to main unless a revision is given.
if [ -n "${BONSAI2_DRAFTER:-}" ]; then
  DRAFTER_REVISION=${BONSAI2_DRAFTER_REVISION:-main}
else
  DRAFTER_REVISION=${BONSAI2_DRAFTER_REVISION:-2a2c2c1e25e82173729bdf104379ffc50d4222c3}
fi

(cd envs/dspark && UV_PROJECT_ENVIRONMENT="$VENV" uv sync --locked --quiet)

# A local directory is used as it is; a Hub repo is fetched at its pinned revision, so the
# server starts from local files and never depends on the network while loading.
fetch() {
  if [ -d "$1" ]; then
    echo "$1"
  else
    "$VENV/bin/python" -c 'import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2]))' "$1" "$2"
  fi
}
TARGET_DIR=$(fetch "$TARGET" "$TARGET_REVISION")
DRAFTER_DIR=$(fetch "$DRAFTER" "$DRAFTER_REVISION")
echo "target  $TARGET@$TARGET_REVISION" >&2
echo "drafter $DRAFTER@$DRAFTER_REVISION" >&2

exec "$VENV/bin/python" bin/mlx-dspark-patched serve \
  --model "$TARGET_DIR" --mode dflash \
  --drafter "$DRAFTER_DIR" --drafter-bits 4 --kv-bits 8 \
  --context-window 131072 --cpu-split 0 \
  --default-temperature 1.0 --default-top-p 0.95 --default-top-k 20 \
  --host "${BONSAI2_HOST:-127.0.0.1}" --port "${BONSAI2_PORT:-8088}"
