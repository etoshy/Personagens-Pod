#!/usr/bin/env bash
# Sobe a API (porta 8000) com o worker. Se o processo cair, volta sozinho em 5 s (o job do meio volta pra fila).
#   PORT=8001 bash start.sh        -> outra porta
#   API_KEY=segredo bash start.sh  -> exige a chave (header X-API-Key)
#   PASSOS=40 CFG=4.0              -> padrão de geração (o pedido pode mudar)
#   OFFLOAD=auto|1|0               -> auto: placa < 70 GB divide o modelo com a RAM (mais lento)
#   MOCK=1                         -> sem modelo (testa o caminho todo sem GPU)
#   RAPIDO=1                       -> ⚡ LoRA Lightning (4 passos, ~10× mais rápido; qualidade um pouco menor)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
WS="${WORKSPACE:-/workspace}"
export HF_HOME="${HF_HOME:-$WS/hf}"
export DATA_DIR="${DATA_DIR:-$WS/personagens_data}"
if [ "${RAPIDO:-0}" = "1" ]; then
  export LIGHTNING_REPO="${LIGHTNING_REPO:-lightx2v/Qwen-Image-Edit-2511-Lightning}"
  export LIGHTNING_FILE="${LIGHTNING_FILE:-Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors}"
  echo "⚡ modo rápido: $LIGHTNING_REPO ($LIGHTNING_FILE) — 4 passos"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PY:-$WS/venvs/personagens/bin/python}"
LOGS="${LOGS:-$WS/personagens_logs}"
export PORT="${PORT:-8000}"
mkdir -p "$LOGS" "$DATA_DIR"
[ -x "$PY" ] || { echo "Rode antes: bash $ROOT/setup.sh"; exit 1; }
bash "$ROOT/stop.sh" >/dev/null 2>&1 || true
sleep 1
cd "$ROOT"
nohup bash -c 'while true; do "$@"; rc=$?; echo "[$(date "+%F %T")] saiu (código $rc) — reiniciando em 5 s"; sleep 5; done' \
  _ "$PY" -m app.main >> "$LOGS/api.log" 2>&1 &
echo "API pid $!  (log: $LOGS/api.log)"
if [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "URL: https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net   (cole no editor: 🧍 Personagens no pod)"
else
  echo "API em http://0.0.0.0:${PORT}"
fi
echo "O modelo carrega em segundo plano (minutos). Acompanhe: tail -f $LOGS/api.log  ou  GET /status"
