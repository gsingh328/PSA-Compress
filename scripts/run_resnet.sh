#!/bin/sh

SCRIPT=cifar10_main.py
OUTPUT_FOLDER="./output/cifar10/"

# Reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8

python ${SCRIPT} --student-resnet --kd-wd 5e-4 --disable-tqdm \
    --save "${OUTPUT_FOLDER}/resnet20.bin" > "${OUTPUT_FOLDER}/resnet20_log.txt"
