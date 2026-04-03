#!/usr/bin/env bash
# =============================================================================
# run_eval.sh — Run the full evaluation suite on a trained LoRA checkpoint
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ---- Defaults ---------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-outputs/checkpoints/final}"
BASE_MODEL="${BASE_MODEL:-gpt2}"
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"
LORA_CONFIG="${LORA_CONFIG:-configs/lora.yaml}"
PROCESSED_DATA="${PROCESSED_DATA:-}"
LOG_DIR="outputs/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ---- Parse CLI args ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path)     MODEL_PATH="$2";       shift 2 ;;
    --base_model)     BASE_MODEL="$2";        shift 2 ;;
    --config)         TRAINING_CONFIG="$2";   shift 2 ;;
    --lora_config)    LORA_CONFIG="$2";       shift 2 ;;
    --processed_data) PROCESSED_DATA="$2";    shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_${TIMESTAMP}.log"

echo "============================================================"
echo "  Evaluation Pipeline — $(date)"
echo "============================================================"
echo "  Model path      : $MODEL_PATH"
echo "  Base model      : $BASE_MODEL"
echo "  Log file        : $LOG_FILE"
echo "============================================================"

PYTHONPATH="$PROJECT_ROOT" python src/evaluation/evaluate.py \
  --model_path "$MODEL_PATH" \
  --base_model "$BASE_MODEL" \
  --config "$TRAINING_CONFIG" \
  --lora_config "$LORA_CONFIG" \
  ${PROCESSED_DATA:+--processed_data "$PROCESSED_DATA"} \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "✅ Evaluation complete. Results saved under outputs/eval/"
