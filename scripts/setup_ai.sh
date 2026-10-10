#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$TASK_ROOT"
TASK_AI_CACHE="$TASK_ROOT/workspace/cache/ai"
mkdir -p "$TASK_AI_CACHE/tmp"
export TMPDIR="$TASK_AI_CACHE/tmp"
export SAM2_BUILD_CUDA=0
TASK_PYTHON="${VISIONREFINE_PYTHON:-python3}"
if [ -z "${VISIONREFINE_PYTHON:-}" ] && [ -x "$TASK_ROOT/.venv/bin/python" ]; then
  TASK_PYTHON="$TASK_ROOT/.venv/bin/python"
fi
"$TASK_PYTHON" -c 'import sys; assert sys.version_info >= (3, 10), "Default models require Python 3.10 or newer"'
if [ ! -x "$TASK_AI_CACHE/venv/bin/python" ]; then
  "$TASK_PYTHON" -m venv "$TASK_AI_CACHE/venv"
fi
TASK_AI_PYTHON="$TASK_AI_CACHE/venv/bin/python"
"$TASK_AI_PYTHON" -m pip install --no-cache-dir 'pip>=25' packaging
"$TASK_AI_PYTHON" scripts/install_ai_runtime.py
"$TASK_AI_PYTHON" -m pip install --no-cache-dir -e '.[segmentation,video]' 'transformers==4.57.1' 'huggingface-hub>=0.34,<1' 'accelerate>=1,<2' 'hydra-core==1.3.2' 'iopath==0.1.10' 'scipy>=1.15,<2' tqdm
"$TASK_AI_PYTHON" -m pip install --no-cache-dir --no-build-isolation --no-deps 'git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4'
"$TASK_AI_PYTHON" scripts/download_ai_models.py
printf '\n模型安装完成。启动命令：\nCUDA_VISIBLE_DEVICES=0 %s -m visionrefine.ai_worker --port 8030\n' "$TASK_AI_PYTHON"
