#!/usr/bin/env bash
# Instala (uma vez por volume) — rodar de novo é rápido: só confere.
#   BAIXAR=0 bash setup.sh   -> não baixa os modelos agora (baixam no 1º start)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
WS="${WORKSPACE:-/workspace}"
VENV="$WS/venvs/personagens"
export HF_HOME="${HF_HOME:-$WS/hf}"
mkdir -p "$WS/venvs" "$HF_HOME"

if [ ! -x "$VENV/bin/python" ]; then
  echo "== criando o ambiente em $VENV (reaproveita o torch do template)"
  python3 -m venv --system-site-packages "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install -q -U pip wheel

if ! "$PY" -c "import torch, torchvision; assert torch.cuda.is_available()" 2>/dev/null; then
  CUDA=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' || echo "12.4")
  case "$CUDA" in 12.4*|12.5*) IDX=cu124 ;; 12.6*|12.7*) IDX=cu126 ;; 12.8*|12.9*) IDX=cu128 ;; 13.*) IDX=cu130 ;; *) IDX=cu124 ;; esac
  echo "== instalando torch ($IDX, driver CUDA $CUDA)"
  "$PY" -m pip install -q torch torchvision --index-url "https://download.pytorch.org/whl/$IDX"
fi
"$PY" - <<'PYEOF'
import torch
v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "nenhuma")
if v < (2, 4):
    print("⚠ torch antigo (< 2.4): troque o template por um 'RunPod PyTorch 2.4+' se der erro")
PYEOF

echo "== dependências"
"$PY" -m pip install -q -r "$ROOT/requirements.txt"
# o pipeline do Qwen-Image-Edit-2511 (QwenImageEditPlusPipeline) está no diffusers do GitHub
"$PY" -c "from diffusers import QwenImageEditPlusPipeline" 2>/dev/null || \
  "$PY" -m pip install -q "git+https://github.com/huggingface/diffusers"
"$PY" -c "from diffusers import QwenImageEditPlusPipeline; print('diffusers ok')"

if [ "${BAIXAR:-1}" = "1" ]; then
  echo "== baixando os modelos pro volume (~60 GB na 1ª vez; depois só confere)"
  "$PY" - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
for repo in (os.getenv("QWEN_EDIT_MODEL", "Qwen/Qwen-Image-Edit-2511"), os.getenv("RECORTE_MODEL", "ZhengPeng7/BiRefNet")):
    print("  ", repo, "->", snapshot_download(repo))
PYEOF
fi
echo "✅ pronto. Agora: bash $ROOT/start.sh"
