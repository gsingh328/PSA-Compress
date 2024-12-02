#!/bin/sh

DATASET="cifar10"
SCRIPT="${DATASET}_main.py"

DATASET_DIR="~/icml_data"

# Select the second option if you want to download the dataset
DATASET_DOWNLOAD_FLAG=""
# DATASET_DOWNLOAD_FLAG="--download-dataset"

# Reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8

OUTPUT_FOLDER="./output/${DATASET}/resmlp_layers_4_h_4_embed_96/bspline_65_1e-5"

python ${SCRIPT} --dataset-dir ${DATASET_DIR} ${DATASET_DOWNLOAD_FLAG} \
--teacher-load $TEACHER_MODEL "${OUTPUT_FOLDER}/h_mul_2_model.bin" \
--test-only --teacher-hidden-factor 2 --teacher-nl-act bspline

python ${SCRIPT} --dataset-dir ${DATASET_DIR} ${DATASET_DOWNLOAD_FLAG} \
--teacher-load $TEACHER_MODEL "${OUTPUT_FOLDER}/h_mul_1_model.bin" \
--test-only --teacher-hidden-factor 1 --teacher-nl-act bspline

python ${SCRIPT} --dataset-dir ${DATASET_DIR} ${DATASET_DOWNLOAD_FLAG} \
--teacher-load $TEACHER_MODEL "${OUTPUT_FOLDER}/h_div_2_model.bin" \
--test-only --teacher-hidden-factor 0.5 --teacher-nl-act bspline

python ${SCRIPT} --dataset-dir ${DATASET_DIR} ${DATASET_DOWNLOAD_FLAG} \
--teacher-load $TEACHER_MODEL "${OUTPUT_FOLDER}/h_div_4_model.bin" \
--test-only --teacher-hidden-factor 0.25 --teacher-nl-act bspline
