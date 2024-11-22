#ifndef ACCEL_PSA_COMMON_H
#define ACCEL_PSA_COMMON_H


#include <ap_int.h>
#include "accel_common.h"

#define PSA_IDX_OFFSET 128

#define LERP_PSA_BITS 8
#define LERP_PSA_MSB_BITS 6
#define LERP_PSA_LSB_BITS (LERP_PSA_BITS - LERP_PSA_MSB_BITS)
#define LERP_PSA_N (1<<LERP_PSA_MSB_BITS)

// #define LERP_PSA_LUT_SIZE LERP_PSA_N + 1
#define LERP_PSA_LUT_SIZE LERP_PSA_N

typedef ap_uint<LERP_PSA_MSB_BITS> psa_msb_idx_t;
typedef ap_uint<LERP_PSA_LSB_BITS> psa_lsb_idx_t;

typedef ap_int<LERP_PSA_BITS*2> psa_lerp_t;


typedef ap_int<LERP_PSA_BITS> model_psa_t;
typedef ap_int<LERP_PSA_BITS + 1> model_psa_off_t;
typedef ap_uint<LERP_PSA_BITS> model_psa_idx_t;


// Data type used at port-level of kernel
typedef hls::vector<model_psa_t, PORT_VEC_SIZE> accel_psa_lut_vec_t;

#define PSA_LUT_BUFF_N 2

#define MAX_PSA_LUT_BUFFER (MAX_I_DIM * LERP_PSA_N)
#define PSA_LUT_BUFF_DIM_0 ((MAX_PSA_LUT_BUFFER)/(TILE_SIZE))
#define PSA_LUT_BUFF_DIM_1 (TILE_SIZE)

// #define PSA_LUT_BUFF_DIM_0 MAX_I_DIM
// #define PSA_LUT_BUFF_DIM_1 LERP_PSA_N

#endif // ACCEL_PSA_COMMON_H
