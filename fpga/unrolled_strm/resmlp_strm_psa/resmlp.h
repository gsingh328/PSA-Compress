#ifndef RESMLP_H
#define RESMLP_H

#include "hls_stream.h"
#include "resmlp_types.h"

#define SEQ_N 64
#define SEQ_N_LG2 6
#define EMBED_N 96
#define HIDDEN_N 96
// #define HIDDEN_N 192

#define IN_SEQ_N SEQ_N
#define IN_VEC_N 48

#define OUT_SEQ_N 1
#define OUT_VEC_N 10

// #define OUT_SEQ_N SEQ_N
// #define OUT_VEC_N EMBED_N

// #define OUT_SEQ_N EMBED_N
// #define OUT_VEC_N SEQ_N

// base fifo depth between layers
#define BASE_FIFO_DEPTH 8
// extra padding to add to residual paths (only for debugging)
#define RESIDUAL_DBG_PAD 0


void krnl_resmlp(
	hls::stream<model_act_t> &in,
	hls::stream<model_act_t> &out
);

#endif // RESMLP_H
