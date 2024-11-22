#ifndef RESMLP_CROSSCHANNEL_H
#define RESMLP_CROSSCHANNEL_H


#include "accel_common.h"
#include "resmlp_common.h"


// In this config X_DIM_0 == W_DIM_0
// So for examples A[256,512] Matmul B[512,1024] = C[256,1024]
// then X in memory will be of shape [512,256] and W will be of shape [512, 1024]
// and this gives Y of shape [1024, 256]

// ========================================================================
// Input
// ========================================================================

#define RM_CC_IN_DIM_0 RESMLP_EMBED_DIM 		// input features
#define RM_CC_IN_DIM_1 MAX_B_DIM 				// batch size

#define RM_CC_HIDDEN_DIM RESMLP_HIDDEN_DIM

// ========================================================================
// Fully Connected Linear Layers
// ========================================================================

// All FC offsets will be relative to 0 within this file
// They can be then included and redefined to a proper offset
#define RM_CC_FC_W_BASE_OFFSET 0
#define RM_CC_FC_BS_BASE_OFFSET 0

// ------------------------------------------------------------------------
// FC Layer 1
// ------------------------------------------------------------------------
#define RM_CC_FC1_W_DIM_0 RM_CC_IN_DIM_0 		// input features
#define RM_CC_FC1_W_DIM_1 RM_CC_HIDDEN_DIM 		// output features

#define RM_CC_FC1_W_OFFSET RM_CC_FC_W_BASE_OFFSET
#define RM_CC_FC1_BS_OFFSET RM_CC_FC_BS_BASE_OFFSET

// ------------------------------------------------------------------------
// FC Layer 2
// ------------------------------------------------------------------------
#define RM_CC_FC2_W_DIM_0 RM_CC_FC1_W_DIM_1 	// input features
#define RM_CC_FC2_W_DIM_1 RM_CC_IN_DIM_0 		// output features

#define RM_CC_FC2_W_OFFSET \
(RM_CC_FC1_W_OFFSET + (RM_CC_FC1_W_DIM_0 * RM_CC_FC1_W_DIM_1))

#define RM_CC_FC2_BS_OFFSET \
(RM_CC_FC1_BS_OFFSET + (RM_CC_FC1_W_DIM_1))

// ========================================================================
// Affines
// ========================================================================

// All affine offset will be relative to 0 within this file
// They can be then included and redefined to a proper offset
#define RM_CC_AFF_W_BASE_OFFSET 0
#define RM_CC_AFF_BS_BASE_OFFSET 0

// ------------------------------------------------------------------------
// Residual Path Affine (no transposing)
// ------------------------------------------------------------------------
#define RM_CC_RES_AFF_W_DIM_0 RM_CC_IN_DIM_0 	// input features

#define RM_CC_RES_AFF_W_OFFSET RM_CC_AFF_W_BASE_OFFSET
#define RM_CC_RES_AFF_BS_OFFSET RM_CC_AFF_BS_BASE_OFFSET

// ========================================================================
// Output
// ========================================================================

#define RM_CC_OUT_DIM_0 RM_CC_L1_W_DIM_1 		// output features
#define RM_CC_OUT_DIM_1 RM_CC_IN_DIM_1 			// batch size


#define RM_CC_AFF_OUT_W_OFFSET \
(RM_CC_RES_AFF_W_OFFSET + (RM_CC_RES_AFF_W_DIM_0))

#define RM_CC_AFF_OUT_BS_OFFSET \
(RM_CC_RES_AFF_BS_OFFSET + (RM_CC_RES_AFF_W_DIM_0))

#define RM_CC_FC_W_OUT_W_OFFSET \
(RM_CC_FC2_W_OFFSET + (RM_CC_FC2_W_DIM_0 * RM_CC_FC2_W_DIM_1))

#define RM_CC_FC_BS_OUT_BS_OFFSET \
(RM_CC_FC2_BS_OFFSET + (RM_CC_FC2_W_DIM_1))


#endif // RESMLP_CROSSCHANNEL_H
