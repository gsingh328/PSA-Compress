#ifndef RESMLP_OUTPOOL_H
#define RESMLP_OUTPOOL_H


#include "accel_common.h"
#include "resmlp_common.h"


// In this config X_DIM_0 == W_DIM_0
// So for examples A[256,512] Matmul B[512,1024] = C[256,1024]
// then X in memory will be of shape [512,256] and W will be of shape [512, 1024]
// and this gives Y of shape [1024, 256]

// ========================================================================
// Input
// ========================================================================

#define RM_OUTP_IN_DIM_0 RESMLP_EMBED_DIM 		// input features
#define RM_OUTP_IN_DIM_1 MAX_B_DIM 				// batch size
#define RM_OUTP_IN_DIM_1_LG2 MAX_B_DIM_LG2      // LOG2 of batch size

// ========================================================================
// Fully Connected Linear Layers
// ========================================================================

// All FC offsets will be relative to 0 within this file
// They can be then included and redefined to a proper offset
#define RM_OUTP_FC_W_BASE_OFFSET 0
#define RM_OUTP_FC_BS_BASE_OFFSET 0

// ------------------------------------------------------------------------
// FC Layer 1
// ------------------------------------------------------------------------
#define RM_OUTP_FC_W_DIM_0 RM_OUTP_IN_DIM_0 		// input features
#define RM_OUTP_FC_W_DIM_1 RESMLP_OUT_DIM_0 		// output features
#define RM_OUTP_FC_W_DIM_1_PADDED 16 		        // output features

#define RM_OUTP_FC_W_OFFSET RM_OUTP_FC_W_BASE_OFFSET
#define RM_OUTP_FC_BS_OFFSET RM_OUTP_FC_BS_BASE_OFFSET

// ========================================================================
// Output
// ========================================================================

// #define RM_OUTP_OUT_DIM_0 RESMLP_OUT_DIM_1 		    // output features
// #define RM_OUTP_OUT_DIM_1 RM_OUTP_IN_DIM_1 			// batch size

#define RM_OUTP_FC_W_OUT_W_OFFSET \
(RM_OUTP_FC_W_OFFSET + (RM_OUTP_FC_W_DIM_0 * RM_OUTP_FC_W_DIM_1_PADDED))

#define RM_OUTP_FC_BS_OUT_BS_OFFSET \
(RM_OUTP_FC_BS_OFFSET + (RM_OUTP_FC_W_DIM_1_PADDED))

#endif // RESMLP_OUTPOOL_H
