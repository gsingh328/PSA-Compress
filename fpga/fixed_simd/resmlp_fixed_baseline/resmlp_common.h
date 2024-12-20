#ifndef RESMLP_COMMON_H
#define RESMLP_COMMON_H

#include "accel_common.h"

#define RESMLP_IN_DIM_0 48
#define RESMLP_IN_DIM_1 64

#define RESMLP_EMBED_DIM 96

#if(MAX_I_DIM==192)
#define RESMLP_HIDDEN_DIM 192
#endif

#if(MAX_I_DIM==96)
#define RESMLP_HIDDEN_DIM 96
#endif


#define RESMLP_OUT_DIM_0 10
#define RESMLP_OUT_DIM_1 1

// #define RESMLP_OUT_DIM_0 96
// #define RESMLP_OUT_DIM_1 64

// #define RESMLP_OUT_DIM_0 64
// #define RESMLP_OUT_DIM_1 96

#define RESMLP_OUT_PADDED_DIM_0 10
#define RESMLP_OUT_PADDED_DIM_1 16

// #define RESMLP_OUT_PADDED_DIM_0 96
// #define RESMLP_OUT_PADDED_DIM_1 64

#define BUF_CP_SEQ_MIN_CNT 48
#define BUF_CP_SEQ_MAX_CNT 96

#define AFFINE_VEC_MIN_CNT ((96*64)/16)
#define AFFINE_VEC_MAX_CNT ((96*64)/16)

#define BUF_CP_PAR_MIN_CNT ((1*64)/16)
#define BUF_CP_PAR_MAX_CNT ((96*96)/16)

#endif // RESMLP_COMMON_H
