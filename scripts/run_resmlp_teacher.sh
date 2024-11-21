#!/bin/sh

SCRIPT=cifar10_main.py
# TEACHER_MODEL="./output/cifar10/resnet20_model.bin"
TEACHER_MODEL="./output/cifar10/resnet110_model.bin"
OUTPUT_FOLDER="./output/cifar10/resmlp_layers_4_h_4_embed_96"

# Reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p ${OUTPUT_FOLDER}

python ${SCRIPT} --teacher-load $TEACHER_MODEL --teacher-resnet --do-kd \
    --student-nl-act base --student-hidden-factor 4 \
    --disable-tqdm --save "${OUTPUT_FOLDER}/teacher_model.bin" > "${OUTPUT_FOLDER}/teacher_model_log.txt"

# python ${SCRIPT} --teacher-load $TEACHER_MODEL --do-kd --hidden-factor 4 --teacher-resnet \
#     --disable-tqdm --save "${OUTPUT_FOLDER}/teacher_model_gelu.bin" > "${OUTPUT_FOLDER}/teacher_model_gelu_log.txt"

# python ${SCRIPT} --teacher-load $TEACHER_MODEL --do-kd --hidden-factor 4 --teacher-resnet \
#     --disable-tqdm --save "${OUTPUT_FOLDER}/teacher_model_silu.bin" > "${OUTPUT_FOLDER}/teacher_model_silu_log.txt"
