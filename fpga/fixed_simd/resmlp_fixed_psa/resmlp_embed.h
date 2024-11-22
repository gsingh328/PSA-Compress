#ifndef RESMLP_EMBED_H
#define RESMLP_EMBED_H


#include "accel_common.h"
#include "resmlp_common.h"


// In this config X_DIM_0 == W_DIM_0
// So for examples A[256,512] Matmul B[512,1024] = C[256,1024]
// then X in memory will be of shape [512,256] and W will be of shape [512, 1024]
// and this gives Y of shape [1024, 256]

// ------------------------------------------------------------------------
// Input
// ------------------------------------------------------------------------

#define RM_EM_IN_DIM_0 RESMLP_IN_DIM_0          // input features
#define RM_EM_IN_DIM_1 MAX_B_DIM 			    // batch size

#define RM_EM_HIDDEN_DIM RESMLP_EMBED_DIM

// ------------------------------------------------------------------------
// Embed Layer 0
// ------------------------------------------------------------------------
#define RM_EM_L0_W_DIM_0 RM_EM_IN_DIM_0 		// input features
#define RM_EM_L0_W_DIM_1 RM_EM_HIDDEN_DIM	    // output features
#define RM_EM_L0_W_OFFSET 0
#define RM_EM_L0_BS_OFFSET 0

// ------------------------------------------------------------------------
// Output
// ------------------------------------------------------------------------

#define RM_EM_OUT_DIM_0 RM_EM_L0_W_DIM_1 		// output features
#define RM_EM_OUT_DIM_1 RM_EM_IN_DIM_1 		    // batch size

#define RM_EM_OUT_W_OFFSET \
(RM_EM_L0_W_OFFSET + (RM_EM_L0_W_DIM_0 * RM_EM_L0_W_DIM_1))

#define RM_EM_OUT_BS_OFFSET \
(RM_EM_L0_BS_OFFSET + (RM_EM_L0_W_DIM_1))

#endif // RESMLP_EMBED_H
