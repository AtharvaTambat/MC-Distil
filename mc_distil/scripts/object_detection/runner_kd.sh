#!/bin/bash
DATA_PATH="./coco"

GPUS="0,4,5,6"
WORKERS=8

STUDENT_BACKBONE="resnet18"
TEACHER_BACKBONE="resnet50"

EPOCHS=12
BATCH_SIZE=16
LEARNING_RATE=0.02
WEIGHT_DECAY=0.0001

ALPHA_KD=0.5
TEMPERATURE=2.0

STUDENT_WEIGHTS_PATH="PATH TO STUDENT CHECKPOINT"
TEACHER_WEIGHTS_PATH="PATH TO TEACHER CHECKPOINT"

python3 mc_distil.training.object_detection.train_kd \
    --data_path ${DATA_PATH} \
    --gpus \"${GPUS}\" \
    --workers ${WORKERS} \
    --student_backbone ${STUDENT_BACKBONE} \
    --teacher_backbone ${TEACHER_BACKBONE} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --wd ${WEIGHT_DECAY} \
    --alpha_kd ${ALPHA_KD} \
    --temperature ${TEMPERATURE}