#ifndef RESMLP_CROSSCHANNEL_PSA_H
#define RESMLP_CROSSCHANNEL_PSA_H


#include "accel_common.h"
#include "accel_psa_common.h"
#include "resmlp_common.h"


// ========================================================================
// Input
// ========================================================================

#define RM_CC_PSA_EMBED_DIM RESMLP_EMBED_DIM
#define RM_CC_PSA_HIDDEN_DIM RESMLP_HIDDEN_DIM

// ========================================================================
// PSA Layers
// ========================================================================

// All PSA offsets will be relative to 0 within this file
// They can be then included and redefined to a proper offset
#define RM_CC_PSA_LUT_BASE_OFFSET 0

// ------------------------------------------------------------------------
// PSA Layer 1
// ------------------------------------------------------------------------
#define RM_CC_PSA1_LUT_DIM_0 RESMLP_EMBED_DIM       // input features
#define RM_CC_PSA1_LUT_DIM_1 LERP_PSA_N 		    // lookput table size

#define RM_CC_PSA1_LUT_OFFSET RM_CC_PSA_LUT_BASE_OFFSET

// ------------------------------------------------------------------------
// PSA Layer 1
// ------------------------------------------------------------------------
#define RM_CC_PSA2_LUT_DIM_0 RESMLP_HIDDEN_DIM      // input features
#define RM_CC_PSA2_LUT_DIM_1 LERP_PSA_N 		    // lookput table size

#define RM_CC_PSA2_LUT_OFFSET \
(RM_CC_PSA1_LUT_OFFSET + (RM_CC_PSA1_LUT_DIM_0 * RM_CC_PSA1_LUT_DIM_1))

// ========================================================================
// Output
// ========================================================================

#define RM_CC_PSA_OUT_LUT_OFFSET \
(RM_CC_PSA2_LUT_OFFSET + (RM_CC_PSA2_LUT_DIM_0 * RM_CC_PSA2_LUT_DIM_1))


#endif // RESMLP_CROSSCHANNEL_PSA_H
