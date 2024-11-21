#!/bin/sh

DATASET="svhn"
SCRIPT="${DATASET}_main.py"

DATASET_DIR="~/icml_data"

# Select the second option if you want to download the dataset
DATASET_DOWNLOAD_FLAG=""
# DATASET_DOWNLOAD_FLAG="--download-dataset"

# Reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PARENT_FOLDER="resmlp_layers_2_h_4_embed_64"

TEACHER_MODEL="./output/${DATASET}/${PARENT_FOLDER}/teacher_model.bin"

# OUTPUT_FOLDER="./output/${DATASET}/${PARENT_FOLDER}/psa"
OUTPUT_FOLDER="./output/${DATASET}/${PARENT_FOLDER}/lerp_psa_64"
mkdir -p ${OUTPUT_FOLDER}

python ${SCRIPT} --dataset-dir ${DATASET_DIR} ${DATASET_DOWNLOAD_FLAG} \
--teacher-load $TEACHER_MODEL --do-kt --do-kd --student-hidden-factor 2 --student-nl-act psa \
--disable-tqdm \
--save "${OUTPUT_FOLDER}/h_mul_2_model.bin" > "${OUTPUT_FOLDER}/h_mul_2_log.txt" &

python ${SCRIPT} --dataset-dir ${DATASET_DIR} ${DATASET_DOWNLOAD_FLAG} \
--teacher-load $TEACHER_MODEL --do-kt --do-kd --student-hidden-factor 1 --student-nl-act psa \
--disable-tqdm \
--save "${OUTPUT_FOLDER}/h_mul_1_model.bin" > "${OUTPUT_FOLDER}/h_mul_1_log.txt"

python ${SCRIPT} --dataset-dir ${DATASET_DIR} ${DATASET_DOWNLOAD_FLAG} \
--teacher-load $TEACHER_MODEL --do-kt --do-kd --student-hidden-factor 0.5 --student-nl-act psa \
--disable-tqdm \
--save "${OUTPUT_FOLDER}/h_div_2_model.bin" > "${OUTPUT_FOLDER}/h_div_2_log.txt" &

python ${SCRIPT} --dataset-dir ${DATASET_DIR} ${DATASET_DOWNLOAD_FLAG} \
--teacher-load $TEACHER_MODEL --do-kt --do-kd --student-hidden-factor 0.25 --student-nl-act psa \
--disable-tqdm \
--save "${OUTPUT_FOLDER}/h_div_4_model.bin" > "${OUTPUT_FOLDER}/h_div_4_log.txt"
