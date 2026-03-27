#pragma once

// TurboQuant CUDA — WHT rotation + quantization for turbo3/turbo4 KV cache
// Port of Metal implementation from ggml-metal.metal

#include "common.cuh"
#include "ggml-common.h"

// ============================================================================
// Constants
// ============================================================================

// 3-bit centroids for d=128 (scaled by 1/sqrt(128))
static const __device__ float d_turbo_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// Midpoints for 3-bit nearest centroid lookup
static const __device__ float d_turbo_mid_3bit[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f
};

// WHT rotation sign arrays (seed=42)
static const __device__ float d_turbo_wht_signs1[128] = {
    -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,
     1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
     1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
     1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
     1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
     1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
     1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f
};

static const __device__ float d_turbo_wht_signs2[128] = {
     1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,
     1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f,
     1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,
     1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
     1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
     1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
     1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f
};

// QJL WHT rotation sign arrays (seed=1042)
static const __device__ float d_turbo_qjl_wht_signs1[128] = {
     1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
     1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
     1.0f, -1.0f,  1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
     1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,
     1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
     1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,
     1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,
     1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f
};

static const __device__ float d_turbo_qjl_wht_signs2[128] = {
     1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,  1.0f, -1.0f, -1.0f,  1.0f,  1.0f,
     1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f,
     1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f, -1.0f,  1.0f,  1.0f, -1.0f,
     1.0f, -1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,
    -1.0f,  1.0f,  1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,
    -1.0f, -1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
    -1.0f,  1.0f, -1.0f,  1.0f,  1.0f,  1.0f,  1.0f, -1.0f
};

// ============================================================================
// Device functions
// ============================================================================

// Fast Walsh-Hadamard Transform (in-place, normalized by 1/sqrt(128))
// O(n log n) = 896 ops for n=128, vs O(n^2) = 16384 for dense matvec
static __device__ void turbo_fwht_128(float * x) {
    #pragma unroll
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    #pragma unroll
    for (int i = 0; i < 128; i++) {
        x[i] *= inv_sqrt_128;
    }
}

// Forward rotation: signs1 -> FWHT -> signs2
static __device__ void turbo_rotate_forward(float * x,
                                            const float * __restrict__ s1,
                                            const float * __restrict__ s2) {
    #pragma unroll
    for (int i = 0; i < 128; i++) x[i] *= s1[i];
    turbo_fwht_128(x);
    #pragma unroll
    for (int i = 0; i < 128; i++) x[i] *= s2[i];
}

// Find nearest 3-bit centroid index using midpoint thresholds
static __device__ __forceinline__ uint8_t turbo_nearest_3bit(float val) {
    if (val < d_turbo_mid_3bit[0]) return 0;
    if (val < d_turbo_mid_3bit[1]) return 1;
    if (val < d_turbo_mid_3bit[2]) return 2;
    if (val < d_turbo_mid_3bit[3]) return 3;
    if (val < d_turbo_mid_3bit[4]) return 4;
    if (val < d_turbo_mid_3bit[5]) return 5;
    if (val < d_turbo_mid_3bit[6]) return 6;
    return 7;
}

// ============================================================================
// Turbo3 group quantization: 128 floats -> 4 blocks of block_turbo3_0
//
// Matches Metal kernel_set_rows_turbo:
// 1. Compute L2 norm across 128 elements
// 2. Normalize
// 3. WHT rotation (signs1 -> FWHT -> signs2)
// 4. Quantize each 32-element sub-block to 3-bit centroids
// 5. Norm correction: store corrected_norm = grp_norm / ||centroid_vector||
// ============================================================================
static __device__ void quantize_f32_turbo3_0_group(const float * __restrict__ src,
                                                   block_turbo3_0 * __restrict__ dst) {
    // Step 1: L2 norm across 128 elements
    float norm_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < 128; j++) {
        norm_sq += src[j] * src[j];
    }
    float grp_norm = sqrtf(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    // Step 2: Normalize and load into local buffer
    float x[128];
    #pragma unroll
    for (int j = 0; j < 128; j++) {
        x[j] = src[j] * inv_norm;
    }

    // Step 3: WHT rotation
    turbo_rotate_forward(x, d_turbo_wht_signs1, d_turbo_wht_signs2);

    // Step 4: Quantize 4 sub-blocks of 32 elements each
    float recon_norm_sq = 0.0f;

    #pragma unroll
    for (int b = 0; b < 4; b++) {
        block_turbo3_0 * blk = &dst[b];
        const int off = b * QK_TURBO3;

        // Clear packed fields
        #pragma unroll
        for (int j = 0; j < QK_TURBO3 / 4; j++) blk->qs[j] = 0;
        #pragma unroll
        for (int j = 0; j < QK_TURBO3 / 8; j++) blk->signs[j] = 0;

        // Quantize rotated values to 3-bit centroids
        #pragma unroll
        for (int j = 0; j < QK_TURBO3; j++) {
            float rv = x[off + j];
            uint8_t idx = turbo_nearest_3bit(rv);

            // Pack: low 2 bits -> qs, high bit -> signs
            blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
            if (idx & 0x4) {
                blk->signs[j / 8] |= (1 << (j % 8));
            }

            // Accumulate centroid reconstruction norm
            float c = d_turbo_centroids_3bit[idx];
            recon_norm_sq += c * c;
        }
    }

    // Step 5: Norm correction — store corrected norm so dequant restores exact L2 norm
    float recon_norm = sqrtf(recon_norm_sq);
    float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
    half corrected_norm_h = __float2half(corrected_norm);

    #pragma unroll
    for (int b = 0; b < 4; b++) {
        dst[b].norm = corrected_norm_h;
    }
}

// ============================================================================
// Turbo4 block quantization: 128 floats -> 1 block_turbo4_0
//
// Matches Metal quantize_turbo4_0:
// 1. Compute L2 norm
// 2. Normalize, keep copy
// 3. WHT rotation
// 4. 3-bit quantization (packed as 3 bits per element)
// 5. Compute residual = normalized - recon (in mixed space, no inverse WHT)
// 6. QJL WHT on residual -> store sign bits
// ============================================================================
static __device__ void quantize_f32_turbo4_0_block(const float * __restrict__ src,
                                                   block_turbo4_0 * __restrict__ dst) {
    // Step 1: L2 norm
    float norm_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < 128; j++) {
        norm_sq += src[j] * src[j];
    }
    float norm = sqrtf(norm_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    dst->norm = __float2half(norm);

    // Step 2: Normalize, keep copy for residual
    float x[128];
    float normalized[128];
    #pragma unroll
    for (int j = 0; j < 128; j++) {
        float v = src[j] * inv_norm;
        x[j] = v;
        normalized[j] = v;
    }

    // Step 3: WHT rotation
    turbo_rotate_forward(x, d_turbo_wht_signs1, d_turbo_wht_signs2);

    // Step 4: 3-bit quantization with bit-packing
    // Clear packed fields
    #pragma unroll
    for (int j = 0; j < QK_TURBO4 * 3 / 8; j++) dst->qs[j] = 0;
    #pragma unroll
    for (int j = 0; j < QK_TURBO4 / 8; j++) dst->signs[j] = 0;

    float recon[128];
    #pragma unroll
    for (int j = 0; j < 128; j++) {
        uint8_t idx = turbo_nearest_3bit(x[j]);
        recon[j] = d_turbo_centroids_3bit[idx];

        // Pack 3-bit index (bit-packed, not 2+1 like turbo3)
        int bit_offset = j * 3;
        int byte_idx   = bit_offset / 8;
        int bit_pos    = bit_offset % 8;
        dst->qs[byte_idx] |= (uint8_t)((idx & 0x7) << bit_pos);
        if (bit_pos > 5 && byte_idx + 1 < QK_TURBO4 * 3 / 8) {
            dst->qs[byte_idx + 1] |= (uint8_t)((idx & 0x7) >> (8 - bit_pos));
        }
    }

    // Step 5: Residual = normalized - recon (mixed space, no inverse WHT)
    float rnorm_sq = 0.0f;
    #pragma unroll
    for (int j = 0; j < 128; j++) {
        x[j] = normalized[j] - recon[j];
        rnorm_sq += x[j] * x[j];
    }
    dst->rnorm = __float2half(sqrtf(rnorm_sq));

    // Step 6: QJL WHT on residual -> sign bits
    turbo_rotate_forward(x, d_turbo_qjl_wht_signs1, d_turbo_qjl_wht_signs2);
    #pragma unroll
    for (int i = 0; i < 128; i++) {
        if (x[i] >= 0.0f) {
            dst->signs[i / 8] |= (1 << (i % 8));
        }
    }
}
