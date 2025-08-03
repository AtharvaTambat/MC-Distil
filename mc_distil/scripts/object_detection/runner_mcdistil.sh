#!/bin/bash
DATA_PATH="./coco"

STUDENT18_CKPT="/mnt/nas/atharvatambat/Multinet-main/coco_checkpoints_resnet18/epoch=11-step=21996.ckpt"
STUDENT34_CKPT="/mnt/nas/atharvatambat/Multinet-main/coco_checkpoints_resnet34/epoch=11-step=21996.ckpt"
TEACHER_CKPT="/mnt/nas/atharvatambat/Multinet-main/coco_checkpoints_resnet50/epoch=11-step=21996.ckpt"

GPUS="0,1,2,3,4"

echo "Starting training with the following configuration:"
echo "Student18 Checkpoint: $STUDENT18_CKPT"
echo "Student34 Checkpoint: $STUDENT34_CKPT"
echo "Teacher Checkpoint:   $TEACHER_CKPT"
echo "Dataset Path:         $DATA_PATH"
echo "GPUs:                 $GPUS"
echo "----------------------------------------------------"

python mc_distil.training.object_detection.train_mcdistil \
    --data_path "$DATA_PATH" \
    --student18_ckpt "$STUDENT18_CKPT" \
    --student34_ckpt "$STUDENT34_CKPT" \
    --teacher_ckpt "$TEACHER_CKPT" \
    --gpus "$GPUS" \
    --batch_size 4 \
    --workers 8 \
    --epochs 24 \
    --lr 1e-4 \
    --meta_lr 1e-5 \
    --meta_interval 10 \
    --temperature 2.0 \
    --grad_clip 0.5 \
    --weight_decay 1e-4

echo "Training finished."