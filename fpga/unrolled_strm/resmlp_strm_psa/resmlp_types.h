#ifndef RESMLP_TYPES_H
#define RESMLP_TYPES_H


#include <ap_int.h>

typedef ap_int<8> model_act_t;
typedef ap_int<8> model_w_t;
typedef ap_int<8> model_b_t;
typedef ap_uint<8> model_s_t;

// max required to store is 128 x 128 x 192
typedef ap_int<23> matmul_acc_t;

// 8-bit x 8-bit
typedef ap_int<16> affine_acc_t;

// typedef ap_int<32> scale_acc_t;
typedef ap_int<9> add_clip_acc_t;

// for accumulating across the 64 sequence length
typedef ap_int<14> pool_acc_t;

// Use for saturating output
#define ACT_MIN -128
#define ACT_MAX 127


#endif // RESMLP_TYPES_H