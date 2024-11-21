WEIGHT_BITS=8
ACT_BITS=8

BASE_FOLDER="./output/cifar10/resmlp_layers_4_h_4_embed_96"
TEACHER_MODEL="${BASE_FOLDER}/parent_model.bin"

# NL_ACT=""
NL_ACT="--nl-act psa"

# STUDENT_FOLDER="${BASE_FOLDER}/baseline"
STUDENT_FOLDER="${BASE_FOLDER}/lerp_psa/main/lerp_psa_8_64_epsilon_1e-0"
SAVE_FOLDER="${STUDENT_FOLDER}/quant_test"

# mkdir -p ${SAVE_FOLDER}

# MODEL_NAME="h_mul_2"
# python cifar10_quant_validate.py \
#     --hidden-factor 2 ${NL_ACT} \
#     --load "${STUDENT_FOLDER}/${MODEL_NAME}_model.bin" \
#     --quantize \
#     --input-bits ${ACT_BITS} \
#     --weight-bits ${WEIGHT_BITS} \
#     --bias-bits ${WEIGHT_BITS} \
#     --output-bits ${ACT_BITS} \
#     --quant-load "${SAVE_FOLDER}/${MODEL_NAME}_quant_a${ACT_BITS}_w${WEIGHT_BITS}_lsq_model.bin" \
#     --dump-params


# MODEL_NAME="h_mul_1"
# python cifar10_quant_validate.py \
#     --hidden-factor 1 ${NL_ACT} \
#     --load "${STUDENT_FOLDER}/${MODEL_NAME}_model.bin" \
#     --quantize \
#     --input-bits ${ACT_BITS} \
#     --weight-bits ${WEIGHT_BITS} \
#     --bias-bits ${WEIGHT_BITS} \
#     --output-bits ${ACT_BITS} \
#     --quant-load "${SAVE_FOLDER}/${MODEL_NAME}_quant_a${ACT_BITS}_w${WEIGHT_BITS}_lsq_model.bin" \
#     --dump-params


MODEL_NAME="h_div_2"
python cifar10_quant_validate.py \
    --hidden-factor 0.5 ${NL_ACT} \
    --load "${STUDENT_FOLDER}/${MODEL_NAME}_model.bin" \
    --quantize \
    --input-bits ${ACT_BITS} \
    --weight-bits ${WEIGHT_BITS} \
    --bias-bits ${WEIGHT_BITS} \
    --output-bits ${ACT_BITS} \
    --quant-load "${SAVE_FOLDER}/${MODEL_NAME}_quant_a${ACT_BITS}_w${WEIGHT_BITS}_lsq_model.bin" \
    --dump-params
