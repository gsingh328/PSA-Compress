#!/bin/sh

SCRIPT=svhn_main.py
# TEACHER_MODEL="./output/svhn/resnet20_model.bin"
OUTPUT_FOLDER="./output/svhn/resmlp_layers_2_h_4_embed_64"

# Reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8

mkdir -p ${OUTPUT_FOLDER}

# python ${SCRIPT} \
#     --student-nl-act base --student-hidden-factor 4 --kd-wd 1e-5 \
#     --disable-tqdm --save "${OUTPUT_FOLDER}/teacher_model.bin" > "${OUTPUT_FOLDER}/teacher_model_log.txt"

# python ${SCRIPT} \
#     --student-nl-act base --student-hidden-factor 4 --kd-wd 1e-5 \
#     --disable-tqdm --save "${OUTPUT_FOLDER}/teacher_model_gelu.bin" > "${OUTPUT_FOLDER}/teacher_model_gelu_log.txt"

python ${SCRIPT} \
    --student-nl-act base --student-hidden-factor 4 --kd-wd 1e-5 \
    --disable-tqdm --save "${OUTPUT_FOLDER}/teacher_model_silu.bin" > "${OUTPUT_FOLDER}/teacher_model_silu_log.txt"
