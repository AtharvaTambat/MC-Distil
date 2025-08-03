#!/bin/bash
# The process is: T -> TA1 -> TA2 -> ... -> Student

echo "=========================================================="
echo "  Starting TAKD   "
echo "=========================================================="

DATA_DIR="./coco"
CHECKPOINT_DIR="distillation_chain_checkpoints_takd"

# Define model architectures
CHAIN_ARCHS=(
    "resnet50"      # Teacher
    "resnet34"      # TA
    "resnet18"      # Student
)

# Define pretrained paths of models
CHAIN_INITIAL_CKPTS=(
    "PATH TO TEACHER CHECKPOINT"
    "PATH TO TA CHECKPOINT"
    "PATH TO STUDENT CHECKPOINT"
)

GPUS="1,3,4"
BATCH_SIZE=16
LEARNING_RATE=0.002
WORKERS=8
EPOCHS=100

current_teacher_ckpt="${CHAIN_INITIAL_CKPTS[0]}"

# Loop through each distillation step in the chain.
for (( i=1; i < ${#CHAIN_ARCHS[@]}; i++ )); do
    teacher_arch="${CHAIN_ARCHS[$i-1]}"
    student_arch="${CHAIN_ARCHS[$i]}"
    
    student_initial_ckpt="${CHAIN_INITIAL_CKPTS[$i]}"
    step_output_dir="${CHECKPOINT_DIR}/step_${i}_${teacher_arch}_to_${student_arch}"
    
    python3 mc_distil.training.object_detection.train_takd \
        --data_path "$DATA_DIR" \
        --output_dir "$step_output_dir" \
        --gpus "$GPUS" \
        --teacher_backbone "$teacher_arch" \
        --student_backbone "$student_arch" \
        --teacher_weights_path "$current_teacher_ckpt" \
        --student_weights_path "$student_initial_ckpt" \
        --batch_size $BATCH_SIZE \
        --lr $LEARNING_RATE \
        --workers $WORKERS \
        --epochs $EPOCHS \

    next_teacher_ckpt=$(ls -v "${step_output_dir}"/*.ckpt | tail -n 1)
    echo -e "\n--- Step ${i} Finished. Best model saved to: ${next_teacher_ckpt} ---"
    
    # Update the teacher checkpoint for the next iteration
    current_teacher_ckpt="$next_teacher_ckpt"

done