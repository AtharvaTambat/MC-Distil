#!/bin/bash

BACKBONE="resnet34"

GPUS="0,1,2,3,4,5"

DATA_PATH="./coco"
EPOCHS=100
BATCH_SIZE=60
WORKERS=8
LEARNING_RATE=0.005
WEIGHT_DECAY=0.0005

# --- Verification ---
if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: COCO data path not found at '$DATA_PATH'"
    exit 1
fi

# --- Execution ---
echo "========================================================"
echo "Starting Pre-training"
echo "Backbone:   $BACKBONE"
echo "Epochs:     $EPOCHS"
echo "Checkpointing: Will save the best model from every 20-epoch interval."
echo "========================================================"

python3 mc_distil.training.object_detection.pretrain \
    --backbone "$BACKBONE" \
    --data_path "$DATA_PATH" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --wd $WEIGHT_DECAY \
    --workers $WORKERS \
    --gpus "$GPUS"

echo "Pre-training finished for $BACKBONE."
