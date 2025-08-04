#!/bin/bash

# STAGE 1: Quantify Sample Difficulty
# STAGE 2: Train a Student Model using Robust Knowledge Distillation

STAGE_TO_RUN=2

GPUS="0,2,4,6"
DATA_PATH="./coco"
TEACHER_CKPT_PATH="PATH TO TEACHER PRETRAINED CHECKPOINT"
STUDENT_CKPT_PATH="PATH TO STUDENT PRETRAINED CHECKPOINT"

STUDENT_MODEL="resnet18"

BATCH_SIZE=16
EPOCHS=12
LEARNING_RATE=0.0001
WEIGHT_DECAY=0.0001
ALPHA=0.3
LAMBDA_KD=0.5
TEMPERATURE=3.0

CMD="python3 train_rmc_coco.py \
    --stage ${STAGE_TO_RUN} \
    --data_path \"${DATA_PATH}\" \
    --gpus \"${GPUS}\""

if [ ${STAGE_TO_RUN} -eq 1 ]; then
    CMD+=" --teacher_ckpt_path \"${TEACHER_CKPT_PATH}\""

elif [ ${STAGE_TO_RUN} -eq 2 ]; then
    CMD+=" --teacher_ckpt_path \"${TEACHER_CKPT_PATH}\""
    CMD+=" --student_ckpt_path \"${STUDENT_CKPT_PATH}\""
    CMD+=" --student_model ${STUDENT_MODEL}"
    CMD+=" --batch_size ${BATCH_SIZE}"
    CMD+=" --epochs ${EPOCHS}"
    CMD+=" --lr ${LEARNING_RATE}"
    CMD+=" --wd ${WEIGHT_DECAY}"
    CMD+=" --alpha ${ALPHA}"
    CMD+=" --lambda_kd ${LAMBDA_KD}"
    CMD+=" --temperature ${TEMPERATURE}"
else
    echo "ERROR: Invalid STAGE_TO_RUN value. Please set it to 1 or 2."
    exit 1
fi
eval ${CMD}