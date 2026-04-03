#!/usr/bin/env bash
# =============================================================================
# run_training.sh — Launch LoRA fine-tuning with optional S3 sync
# =============================================================================
set -euo pipefail

# Resolve project root (parent of this script's directory).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ---- Defaults ---------------------------------------------------------------
TRAINING_CONFIG="${TRAINING_CONFIG:-configs/training.yaml}"
LORA_CONFIG="${LORA_CONFIG:-configs/lora.yaml}"
PROCESSED_DATA="${PROCESSED_DATA:-}"
LOG_DIR="outputs/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

# ---- Parse CLI args ---------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)         TRAINING_CONFIG="$2"; shift 2 ;;
    --lora_config)    LORA_CONFIG="$2";     shift 2 ;;
    --processed_data) PROCESSED_DATA="$2";  shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ---- Sanity checks ----------------------------------------------------------
if ! command -v python &> /dev/null; then
  echo "ERROR: python not found. Please activate your virtual environment."
  exit 1
fi

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

echo "============================================================"
echo "  LoRA Fine-Tuning Pipeline — $(date)"
echo "============================================================"
echo "  Training config : $TRAINING_CONFIG"
echo "  LoRA config     : $LORA_CONFIG"
echo "  Log file        : $LOG_FILE"
echo "============================================================"

# ---- Run training -----------------------------------------------------------
PYTHONPATH="$PROJECT_ROOT" python src/training/train.py \
  --config "$TRAINING_CONFIG" \
  --lora_config "$LORA_CONFIG" \
  ${PROCESSED_DATA:+--processed_data "$PROCESSED_DATA"} \
  2>&1 | tee "$LOG_FILE"

echo ""
echo "✅ Training completed. Artifacts saved under outputs/checkpoints/"
