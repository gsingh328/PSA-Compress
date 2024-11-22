#!/bin/sh

DATASET="cifar10"
SCRIPT="${DATASET}_quant.py"

DATASET_DIR="~/icml_data"

# Select the second option if you want to download the dataset
DATASET_DOWNLOAD_FLAG=""
# DATASET_DOWNLOAD_FLAG="--download-dataset"

# Reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8

LR="1e-4"
EPOCHS="50"
WEIGHT_BITS=8
ACT_BITS=8

BASE_FOLDER="./output/cifar10/resmlp_layers_4_h_4_embed_96"
TEACHER_MODEL="${BASE_FOLDER}/teacher_model.bin"
STUDENT_FOLDER="${BASE_FOLDER}/baseline"

HIDDEN_FACTOR=1

# NL_ACT="base"
NL_ACT="psa"

MODEL_NAME="h_mul_${HIDDEN_FACTOR}"
SAVE_FOLDER="${STUDENT_FOLDER}/quant"

mkdir -p ${SAVE_FOLDER}

python ${SCRIPT} \
    --dataset-dir ${DATASET_DIR} \
    ${DATASET_DOWNLOAD_FLAG} \
    --load "${STUDENT_FOLDER}/${MODEL_NAME}_model.bin" \
    --student-hidden-factor 1 \
    --student-nl-act ${NL_ACT} \
    --quantize \
    --input-bits ${ACT_BITS} \
    --weight-bits ${WEIGHT_BITS} \
    --bias-bits ${WEIGHT_BITS} \
    --output-bits ${ACT_BITS} \
    --quant-load "${SAVE_FOLDER}/${MODEL_NAME}_quant_a${ACT_BITS}_w${WEIGHT_BITS}_lsq_model.bin" \
    --dump-params \
    --dump-dir "${SAVE_FOLDER}"
