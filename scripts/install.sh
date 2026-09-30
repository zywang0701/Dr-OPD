#!/bin/bash
# Build the training environment: a venv with verl + vLLM + flash-attn, then the models.
# Called by "bash run.sh setup"; can also be run directly:
#   WORK=./work VENV=./work/venv bash scripts/install.sh
# The pinned versions are the ones all reported runs used. Linux + CUDA 12.x + python3.11, 8 GPUs.
# Propagate pipeline errors and require the wheel's Python version.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-$HERE/work}
VENV=${VENV:-$WORK/venv}
MODELS=${MODELS:-$WORK/models}
PY=${PYTHON:-python3.11}
IDX=${PIP_INDEX:-https://pypi.org/simple/}
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required; set PYTHON to its executable."'
mkdir -p "$WORK" "$MODELS"
log() { echo "[$(date '+%H:%M:%S')] $*"; }

# 1. uv (fast resolver; plain pip works too if you prefer)
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
[ -x "$UV" ] || { log "installing uv"; curl -fsSL https://astral.sh/uv/install.sh | sh; UV=$HOME/.local/bin/uv; }

# 2. venv + pinned deps
[ -x "$VENV/bin/python" ] || "$UV" venv -q "$VENV" --python "$PY"
log "installing python deps"
"$UV" pip install --python "$VENV/bin/python" --index-url "$IDX" \
  vllm==0.10.2 transformers==4.57.6 "ray[default]" hydra-core "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
  codetiming pylatexenc latex2sympy2_extended math_verify wandb torchdata datasets==5.0.1 "pyarrow>=19.0.0" \
  dill peft accelerate "numpy<2.0.0" pandas pip ninja packaging wheel setuptools psutil 2>&1 | tail -3
"$UV" pip install --python "$VENV/bin/python" --index-url "$IDX" --no-deps -e "$HERE/Dropd/verl" 2>&1 | tail -2

# 3. flash-attn 2.8.3 (verl runs with attn_implementation=flash_attention_2)
fa_ok() {
  "$VENV/bin/python" - <<'PY' 2>&1 | tail -1
import torch, flash_attn
from flash_attn import flash_attn_func
q = torch.randn(1, 256, 8, 64, device="cuda", dtype=torch.bfloat16)
print("flash_attn", flash_attn.__version__, tuple(flash_attn_func(q, q, q, causal=True).shape))
PY
}
if ! fa_ok | grep -q '^flash_attn'; then
  ABI=$("$VENV/bin/python" -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')
  W="flash_attn-2.8.3+cu12torch2.8cxx11abi${ABI}-cp311-cp311-linux_x86_64.whl"
  log "trying the prebuilt wheel $W"
  if curl -fsSL --connect-timeout 10 -m 900 -o "$WORK/$W" \
      "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${W/+/%2B}"; then
    "$UV" pip install --python "$VENV/bin/python" --no-deps --reinstall "$WORK/$W" 2>&1 | tail -1
  fi
fi
if ! fa_ok | grep -q '^flash_attn'; then
  NVCC=$(command -v nvcc || true)
  if [ -z "$NVCC" ]; then
    for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
      if [ -x "$candidate" ]; then NVCC=$candidate; break; fi
    done
  fi
  [ -n "$NVCC" ] || { log "no nvcc: install flash-attn 2.8.3 manually"; exit 1; }
  export CUDA_HOME=$(dirname "$(dirname "$NVCC")"); export PATH=$CUDA_HOME/bin:$PATH
  log "building flash-attn from source (15-20 min)"
  FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-9.0} MAX_JOBS=${MAX_JOBS:-32} NVCC_THREADS=2 \
    "$VENV/bin/python" -m pip install -i "$IDX" --no-cache-dir --no-binary=:all: --no-build-isolation \
    --no-deps --force-reinstall flash-attn==2.8.3 2>&1 | tail -3
fi
log "flash-attn check: $(fa_ok)"

# 4. models (student + teacher); skip by exporting SKIP_MODELS=1 and passing HF ids to run.sh instead
if [ "${SKIP_MODELS:-0}" != 1 ]; then
  log "downloading Qwen3-1.7B and Qwen3-4B into $MODELS"
  HF_HUB_ENABLE_HF_TRANSFER=0 "$VENV/bin/python" - "$MODELS" <<'PY'
import sys
from huggingface_hub import snapshot_download
for repo in ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"):
    d = snapshot_download(repo, local_dir="%s/%s" % (sys.argv[1], repo.split("/")[1]),
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"], max_workers=4)
    print("ok", repo, d, flush=True)
PY
fi
log "done. venv $VENV, models $MODELS"
log "next: set TRAIN_DATASET and TEST_DATASET; bash run.sh data; bash run.sh train dropd 0.4 2"
