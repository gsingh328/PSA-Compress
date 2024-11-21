LR="1e-4"
EPOCHS="50"
WEIGHT_BITS=8
ACT_BITS=8

BASE_FOLDER="./output/cifar10/resmlp_layers_4_h_4_embed_96"
TEACHER_MODEL="${BASE_FOLDER}/parent_model.bin"
# STUDENT_FOLDER="${BASE_FOLDER}/baseline"
# STUDENT_FOLDER="${BASE_FOLDER}/psa"
# STUDENT_FOLDER="${BASE_FOLDER}/psa_8_epsilon_1e-0"
# STUDENT_FOLDER="${BASE_FOLDER}/lerp_psa_8_256_epsilon_1e-0"
STUDENT_FOLDER="${BASE_FOLDER}/lerp_psa/main/lerp_psa_8_64_epsilon_1e-0"


# NL_ACT=""
NL_ACT="--nl-act psa"


SAVE_FOLDER="${STUDENT_FOLDER}/quant_test"

mkdir -p ${SAVE_FOLDER}

# TEACHER_MODEL="./resnet20_model.bin"
# MODEL_NAME="h_mul_4"
# python cifar10_quant_validate.py \
#     --load "${STUDENT_FOLDER}/${MODEL_NAME}_model.bin" \
#     --hidden-factor 4 ${NL_ACT} \
#     --quantize \
#     --input-bits ${ACT_BITS} \
#     --weight-bits ${WEIGHT_BITS} \
#     --bias-bits ${WEIGHT_BITS} \
#     --output-bits ${ACT_BITS} \
#     --qat \
#     --do-kd \
#     --teacher-load ${TEACHER_MODEL} \
#     --kd-epochs ${EPOCHS} \
#     --kd-lr ${LR} \
#     --save "${SAVE_FOLDER}/${MODEL_NAME}_quant_a${ACT_BITS}_w${WEIGHT_BITS}_lsq_model.bin" \
#     --disable-tqdm > ${SAVE_FOLDER}/${MODEL_NAME}_quant_a${ACT_BITS}_w${WEIGHT_BITS}_lsq_log.txt

# MODEL_NAME="h_mul_2"
# python cifar10_quant_validate.py \
#     --load "${STUDENT_FOLDER}/${MODEL_NAME}_model.bin" \
#     --hidden-factor 2 ${NL_ACT} \
#     --quantize \
#     --input-bits ${ACT_BITS} \
#     --weight-bits ${WEIGHT_BITS} \
#     --bias-bits ${WEIGHT_BITS} \
#     --output-bits ${ACT_BITS} \
#     --qat \
#     --do-kd \
#     --teacher-load ${TEACHER_MODEL} \
#     --kd-epochs ${EPOCHS} \
#     --kd-lr ${LR} \
#     --save "${SAVE_FOLDER}/${MODEL_NAME}_quant_a${ACT_BITS}_w${WEIGHT_BITS}_lsq_model.bin" \
#     --disable-tqdm > ${SAVE_FOLDER}/${MODEL_NAME}_quant_a${ACT_BITS}_w${WEIGHT_BITS}_lsq_log.txt

# MODEL_NAME="h_mul_1"
# python cifar10_quant_validate.py \
#     --load "${STUDENT_FOLDER}/${MODEL_NAME}_model.bin" \
#     --hidden-factor 1 ${NL_ACT} \
#     --quantize \
#     --input-bits ${ACT_BITS} \
#     --weight-bits ${WEIGHT_BITS} \
#     --bias-bits ${WEIGHT_BITS} \
#     --output-bits ${ACT_BITS} \
#     --qat \
#     --do-kd \
#     --teacher-load ${TEACHER_MODEL} \
#     --kd-epochs ${EPOCHS} \
#     --kd-lr ${LR} \
#     --save "${SAVE_FOLDER}/${MODEL_NAME}_quant_a${ACT_BITS}_w${WEIGHT_BITS}_lsq_model.bin" \
#     --disable-tqdm > ${SAVE_FOLDER}/${MODEL_NAME}_quant_a${ACT_BITS}_w${WEIGHT_BITS}_lsq_log.txt

MODEL_NAME="h_div_2"
python cifar10_quant_validate.py \
    --load "${STUDENT_FOLDER}/${MODEL_NAME}_model.bin" \
    --hidden-factor 0.5 ${NL_ACT} \
    --quantize \
    --input-bits ${ACT_BITS} \
    --weight-bits ${WEIGHT_BITS} \
    --bias-bits ${WEIGHT_BITS} \
    --output-bits ${ACT_BITS} \
    --qat \
    --do-kd \
    --teacher-load ${TEACHER_MODEL} \
    --kd-epochs ${EPOCHS} \
    --kd-lr ${LR} \
    --save "${SAVE_FOLDER}/${MODEL_NAME}_quant_a${ACT_BITS}_w${WEIGHT_BITS}_lsq_model.bin" \
    --disable-tqdm #> ${SAVE_FOLDER}/${MODEL_NAME}_quant_a${ACT_BITS}_w${WEIGHT_BITS}_lsq_log.txt

# MODEL_NAME="h_div_4"
# python cifar10_quant_validate.py \
#     --load "${STUDENT_FOLDER}/${MODEL_NAME}_model.bin" \
#     --hidden-factor 0.25 ${NL_ACT} \
#     --quantize \
#     --input-bits ${ACT_BITS} \
#     --weight-bits ${WEIGHT_BITS} \
#     --bias-bits ${WEIGHT_BITS} \
#     --output-bits ${ACT_BITS} \
#     --qat \
#     --do-kd \
#     --teacher-load ${TEACHER_MODEL} \
#     --kd-epochs ${EPOCHS} \
#     --kd-lr ${LR} \
#     --save "${SAVE_FOLDER}/${MODEL_NAME}_quant_a${ACT_BITS}_w${WEIGHT_BITS}_lsq_model.bin" \
#     --disable-tqdm > ${SAVE_FOLDER}/${MODEL_NAME}_quant_a${ACT_BITS}_w${WEIGHT_BITS}_lsq_log.txt

# MODEL_NAME="h_div_6"
# python cifar10_quant_validate.py \
#     --load "${STUDENT_FOLDER}/${MODEL_NAME}_model.bin" \
#     --hidden-factor 0.167 ${NL_ACT} \
#     --quantize \
#     --input-bits ${ACT_BITS} \
#     --weight-bits ${WEIGHT_BITS} \
#     --bias-bits ${WEIGHT_BITS} \
#     --output-bits ${ACT_BITS} \
#     --qat \
#     --do-kd \
#     --teacher-load ${TEACHER_MODEL} \
#     --kd-epochs ${EPOCHS} \
#     --kd-lr ${LR} \
#     --save "${SAVE_FOLDER}/${MODEL_NAME}_quant_a${ACT_BITS}_w${WEIGHT_BITS}_lsq_model.bin" \
#     --disable-tqdm > ${SAVE_FOLDER}/${MODEL_NAME}_quant_a${ACT_BITS}_w${WEIGHT_BITS}_lsq_log.txt

# MODEL_NAME="h_div_8"
# python cifar10_quant_validate.py \
#     --load "${STUDENT_FOLDER}/${MODEL_NAME}_model.bin" \
#     --hidden-factor 0.125 ${NL_ACT} \
#     --quantize \
#     --input-bits ${ACT_BITS} \
#     --weight-bits ${WEIGHT_BITS} \
#     --bias-bits ${WEIGHT_BITS} \
#     --output-bits ${ACT_BITS} \
#     --qat \
#     --do-kd \
#     --teacher-load ${TEACHER_MODEL} \
#     --kd-epochs ${EPOCHS} \
#     --kd-lr ${LR} \
#     --save "${SAVE_FOLDER}/${MODEL_NAME}_quant_a${ACT_BITS}_w${WEIGHT_BITS}_lsq_model.bin" \
#     --disable-tqdm > ${SAVE_FOLDER}/${MODEL_NAME}_quant_a${ACT_BITS}_w${WEIGHT_BITS}_lsq_log.txt
