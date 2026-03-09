#!/bin/bash
# Launch LIBERO fine-tuning with torchrun (FSDP).
# Usage: NGPU=4 bash script/run_train_libero.sh
set -euo pipefail

NGPU="${NGPU:-4}"
MASTER_PORT="${MASTER_PORT:-29501}"
CONFIG_NAME="${CONFIG_NAME:-libero_a2a}"
DATA_ROOT="${DATA_ROOT:-data/libero_preprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/libero-a2a-ft}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
TRAIN_STEPS="${TRAIN_STEPS:-4000}"
LR="${LR:-1e-5}"

echo "=== LIBERO Fine-tuning ==="
echo "  GPUs:          $NGPU"
echo "  Config:        $CONFIG_NAME"
echo "  Data root:     $DATA_ROOT"
echo "  Output:        $OUTPUT_DIR"
echo "  Batch size:    $BATCH_SIZE x $NGPU GPUs x $GRAD_ACCUM accum"
echo "  Steps:         $TRAIN_STEPS"
echo "  LR:            $LR"
echo "=========================="

torchrun \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    train_libero.py \
    --config-name "$CONFIG_NAME" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --train-steps "$TRAIN_STEPS" \
    --lr "$LR" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum-steps "$GRAD_ACCUM"
