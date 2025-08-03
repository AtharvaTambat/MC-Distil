#!/bin/bash
DATA_PATH="./coco"

STUDENT_BACKBONE="resnet34"

STUDENT_CKPT="PATH TO STUDENT PRETRAINED CHECKPOINT"
TEACHER_CKPT="PATH TO TEACHER PRETRAINED CHECKPOINT"

GPUS="0,1,2"

echo "Starting training with the following configuration:"
echo "Student Backbone:     $STUDENT_BACKBONE"
echo "Student Checkpoint:   $STUDENT_CKPT"
echo "Teacher Checkpoint:   $TEACHER_CKPT"
echo "Dataset Path:         $DATA_PATH"
echo "GPUs:                 $GPUS"
echo "----------------------------------------------------"

python3 mc_distil.training.object_detection.train_meta \
    --data_path "$DATA_PATH" \
    --student_backbone "$STUDENT_BACKBONE" \
    --student_ckpt "$STUDENT_CKPT" \
    --teacher_ckpt "$TEACHER_CKPT" \
    --gpus "$GPUS" \
    --batch_size 8 \
    --workers 8 \
    --lr 1e-4 \
    --meta_lr 1e-5 \
    --meta_interval 10 \
    --temperature 2.0 \
    --grad_clip 0.5 \
    --weight_decay 1e-4