#!/bin/bash

DATA_DIR="./coco"

STUDENT_ARCH="resnet18"
STUDENT_CKPT="PATH TO STUDENT PRETRAINED MODEL"

TEACHER_ARCH="resnet50"
TEACHER_CKPT="PATH TO TEACHER PRETRAINED MODEL"


TA_ARCHS=(
    "resnet34"
)
TA_CKPTS=(
    "PATH TO TEACHING ASSISTANT TRAINED MODEL"
)

GPUS="0,4" 
BATCH_SIZE=16
LEARNING_RATE=0.001
WORKERS=8

python3 mc_distil.training.object_detection.train_dgkd \
    --data_path "$DATA_DIR" \
    --gpus "$GPUS" \
    --student_backbone "$STUDENT_ARCH" \
    --teacher_backbone "$TEACHER_ARCH" \
    --ta_backbones "${TA_ARCHS[@]}" \
    --student_weights_path "$STUDENT_CKPT" \
    --teacher_weights_path "$TEACHER_CKPT" \
    --ta_weights_paths "${TA_CKPTS[@]}" \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --workers $WORKERS \
    --alpha_kd 0.7 \
    --temperature 5.0 \

echo "------------------------------------"
echo "Training finished."