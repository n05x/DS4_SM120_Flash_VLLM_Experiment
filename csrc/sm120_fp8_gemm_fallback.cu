#include <algorithm>
#include <cstdlib>
#include <cstring>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/python.h>

#include "cute/arch/mma_sm120.hpp"
#include "cutlass/numeric_types.h"

#include "jit_kernels/impls/sm120_fp8_gemm_fallback.hpp"
#include "utils/system.hpp"
#include "jit_kernels/impls/smxx_cublaslt.hpp"
#include "sm120_profile.hpp"
#include "utils/exception.hpp"

namespace deep_gemm {
namespace {

void destroy_if(cublasLtMatmulPreference_t pref) {
    if (pref != nullptr)
        cublasLtMatmulPreferenceDestroy(pref);
}

void destroy_if(cublasLtMatmulDesc_t desc) {
    if (desc != nullptr)
        cublasLtMatmulDescDestroy(desc);
}

void destroy_if(cublasLtMatrixLayout_t layout) {
    if (layout != nullptr)
        cublasLtMatrixLayoutDestroy(layout);
}

__device__ __forceinline__ float fp8_e4m3fn_to_float(uint8_t raw) {
    __nv_fp8_e4m3 value;
    value.__x = raw;
    return static_cast<float>(value);
}

__device__ __forceinline__ float fp4_e2m1_to_float(uint8_t code) {
    const uint8_t value_idx = code & 0x07u;
    float value = 0.0f;
    switch (value_idx) {
        case 0: value = 0.0f; break;
        case 1: value = 0.5f; break;
        case 2: value = 1.0f; break;
        case 3: value = 1.5f; break;
        case 4: value = 2.0f; break;
        case 5: value = 3.0f; break;
        case 6: value = 4.0f; break;
        default: value = 6.0f; break;
    }
    return (code & 0x08u) && value_idx != 0 ? -value : value;
}

__device__ __forceinline__ float load_scale(const float* sf, int64_t row,
                                            int64_t col, int gran_mn,
                                            int gran_k, int64_t stride0,
                                            int64_t stride1) {
    const int64_t sf_row = row / gran_mn;
    const int64_t sf_col = col / gran_k;
    return sf[sf_row * stride0 + sf_col * stride1];
}

__device__ __forceinline__ float scale_from_ue8m0_exponent(int exponent) {
    if (exponent == 0)
        return 0.0f;
    return exp2f(static_cast<float>(exponent) - 127.0f);
}

__device__ __forceinline__ float load_scale_any(const void* sf,
                                                int scale_type,
                                                int64_t row,
                                                int64_t col,
                                                int gran_mn,
                                                int gran_k,
                                                int64_t stride0,
                                                int64_t stride1) {
    const int64_t sf_row = row / gran_mn;
    const int64_t sf_col = col / gran_k;
    if (scale_type == 0) {
        const auto* fp32 = static_cast<const float*>(sf);
        return fp32[sf_row * stride0 + sf_col * stride1];
    }

    const auto* packed = static_cast<const int32_t*>(sf);
    const int32_t word =
        packed[sf_row * stride0 + static_cast<int64_t>(sf_col / 4) * stride1];
    const int exponent = (word >> ((sf_col & 3) * 8)) & 0xff;
    return scale_from_ue8m0_exponent(exponent);
}

	__device__ __forceinline__ float load_vec128_scale(const void* scale,
	                                                   int scale_type,
	                                                   int64_t base,
	                                                   int64_t stride_last,
	                                                   int k_block) {
    if (scale_type == 0) {
        const auto* sf = static_cast<const float*>(scale);
        return sf[base + static_cast<int64_t>(k_block) * stride_last];
    }

    const auto* packed = static_cast<const int32_t*>(scale);
    const int32_t word =
        packed[base + static_cast<int64_t>(k_block / 4) * stride_last];
    const int exponent = (word >> ((k_block & 3) * 8)) & 0xff;
	    return scale_from_ue8m0_exponent(exponent);
	}

	int env_int_or_default(const char* name, int default_value) {
	    const char* value = std::getenv(name);
	    if (value == nullptr)
	        return default_value;
	    const int parsed = std::atoi(value);
	    return parsed > 0 ? parsed : default_value;
	}

    bool env_flag_enabled(const char* name) {
        const char* value = std::getenv(name);
        return value != nullptr && std::strcmp(value, "0") != 0 &&
               std::strcmp(value, "false") != 0 &&
               std::strcmp(value, "False") != 0;
    }

__device__ __forceinline__ uint8_t load_ue8m0_raw_packed(
    const int32_t* scale, int64_t base, int64_t stride_last, int k_block) {
    const int32_t word =
        scale[base + static_cast<int64_t>(k_block / 4) * stride_last];
    return static_cast<uint8_t>((word >> ((k_block & 3) * 8)) & 0xff);
}

__global__ void dequant_fp8_c128_kernel(const __nv_fp8_e4m3* x,
                                        const void* scale,
                                        __nv_bfloat16* out,
                                        int scale_type,
                                        int64_t rows, int64_t cols,
                                        int64_t x_stride0,
                                        int64_t x_stride1,
                                        int64_t scale_stride0,
                                        int64_t scale_stride1,
                                        int gran_mn, int gran_k) {
    const int64_t total = rows * cols;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int64_t row = idx / cols;
        const int64_t col = idx - row * cols;
        const float sf = load_scale_any(scale, scale_type, row, col, gran_mn,
                                        gran_k, scale_stride0, scale_stride1);
        const auto* bytes = reinterpret_cast<const uint8_t*>(x);
        const float value =
            fp8_e4m3fn_to_float(bytes[row * x_stride0 + col * x_stride1]) * sf;
        out[idx] = __float2bfloat16(value);
    }
}

template <typename out_t>
__device__ __forceinline__ void store_direct_gemm(out_t* out, int64_t offset,
                                                  float value) {
    out[offset] = static_cast<out_t>(value);
}

template <int kColsPerBlock>
__device__ __forceinline__ void reduce_cols_in_block(
    float (&partial)[kColsPerBlock],
    float (&warp_sums)[kColsPerBlock][8]) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

#pragma unroll
    for (int i = 0; i < kColsPerBlock; ++i) {
        float value = partial[i];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
        if (lane == 0)
            warp_sums[i][warp] = value;
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (int i = 0; i < kColsPerBlock; ++i) {
            float value = lane < num_warps ? warp_sums[i][lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                value += __shfl_down_sync(0xffffffff, value, offset);
            if (lane == 0)
                warp_sums[i][0] = value;
        }
    }
    __syncthreads();
}

template <typename out_t>
__device__ __forceinline__ float load_direct_gemm(const out_t* out,
                                                  int64_t offset) {
    return static_cast<float>(out[offset]);
}

template <>
__device__ __forceinline__ float load_direct_gemm<__nv_bfloat16>(
    const __nv_bfloat16* out, int64_t offset) {
    return __bfloat162float(out[offset]);
}

template <>
__device__ __forceinline__ void store_direct_gemm<__nv_bfloat16>(
    __nv_bfloat16* out, int64_t offset, float value) {
    out[offset] = __float2bfloat16(value);
}

template <typename out_t>
__global__ void fp8_c128_direct_gemm_kernel(
    const uint8_t* __restrict__ a, const void* __restrict__ sfa,
    const uint8_t* __restrict__ b, const void* __restrict__ sfb,
    out_t* __restrict__ d, int sfa_type, int sfb_type, int m, int n, int k, int64_t a_stride0,
    int64_t a_stride1, int64_t b_stride0, int64_t b_stride1,
    int64_t d_stride0, int64_t d_stride1, int64_t sfa_stride0,
    int64_t sfa_stride1, int64_t sfb_stride0, int64_t sfb_stride1,
    int gran_mn_a, int gran_k_a, int gran_mn_b, int gran_k_b,
    bool accumulate) {
    __shared__ float reductions[256];
    const int linear = blockIdx.x;
    const int row = linear / n;
    const int col = linear - row * n;
    float partial = 0.0f;

    for (int kk = threadIdx.x; kk < k; kk += blockDim.x) {
        const float av = fp8_e4m3fn_to_float(
            a[static_cast<int64_t>(row) * a_stride0 +
              static_cast<int64_t>(kk) * a_stride1]);
        const float bv = fp8_e4m3fn_to_float(
            b[static_cast<int64_t>(col) * b_stride0 +
              static_cast<int64_t>(kk) * b_stride1]);
        const float as = load_scale_any(sfa, sfa_type, row, kk, gran_mn_a,
                                        gran_k_a, sfa_stride0, sfa_stride1);
        const float bs = load_scale_any(sfb, sfb_type, col, kk, gran_mn_b,
                                        gran_k_b, sfb_stride0, sfb_stride1);
        partial += av * bv * as * bs;
    }

    reductions[threadIdx.x] = partial;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reductions[threadIdx.x] += reductions[threadIdx.x + offset];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const int64_t out_offset =
            static_cast<int64_t>(row) * d_stride0 +
            static_cast<int64_t>(col) * d_stride1;
        float value = reductions[0];
        if (accumulate)
            value += load_direct_gemm<out_t>(d, out_offset);
        store_direct_gemm<out_t>(d, out_offset, value);
    }
}

template <typename out_t, int kColsPerBlock>
__global__ void fp8_c128_m1_multi_col_gemm_kernel(
    const uint8_t* __restrict__ a, const float* __restrict__ sfa,
    const uint8_t* __restrict__ b, const float* __restrict__ sfb,
    out_t* __restrict__ d, int n, int k, int64_t a_stride0,
    int64_t a_stride1, int64_t b_stride0, int64_t b_stride1,
    int64_t d_stride0, int64_t d_stride1, int64_t sfa_stride0,
    int64_t sfa_stride1, int64_t sfb_stride0, int64_t sfb_stride1,
    int gran_mn_a, int gran_k_a, int gran_mn_b, int gran_k_b,
    bool accumulate) {
    __shared__ float reductions[kColsPerBlock][8];
    float partial[kColsPerBlock];
#pragma unroll
    for (int i = 0; i < kColsPerBlock; ++i)
        partial[i] = 0.0f;

    const int col_base = blockIdx.x * kColsPerBlock;
    for (int kk = threadIdx.x; kk < k; kk += blockDim.x) {
        const float av = fp8_e4m3fn_to_float(
            a[static_cast<int64_t>(kk) * a_stride1]);
        const float as = load_scale(sfa, 0, kk, gran_mn_a, gran_k_a,
                                    sfa_stride0, sfa_stride1);
        const float a_scaled = av * as;
#pragma unroll
        for (int i = 0; i < kColsPerBlock; ++i) {
            const int col = col_base + i;
            if (col < n) {
                const float bv = fp8_e4m3fn_to_float(
                    b[static_cast<int64_t>(col) * b_stride0 +
                      static_cast<int64_t>(kk) * b_stride1]);
                const float bs = load_scale(sfb, col, kk, gran_mn_b, gran_k_b,
                                            sfb_stride0, sfb_stride1);
                partial[i] += a_scaled * bv * bs;
            }
        }
    }

    reduce_cols_in_block<kColsPerBlock>(partial, reductions);

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < kColsPerBlock; ++i) {
            const int col = col_base + i;
            if (col < n) {
                const int64_t out_offset =
                    static_cast<int64_t>(col) * d_stride1;
                float value = reductions[i][0];
                if (accumulate)
                    value += load_direct_gemm<out_t>(d, out_offset);
                store_direct_gemm<out_t>(d, out_offset, value);
            }
        }
    }
}

__global__ void fp8_c128_dequant_a_m1_kernel(
    const uint8_t* __restrict__ a, const void* __restrict__ sfa,
    float* __restrict__ a_dequant, int sfa_type, int k, int64_t a_stride1,
    int64_t sfa_stride0, int64_t sfa_stride1, int gran_k_a) {
    for (int kk = blockIdx.x * blockDim.x + threadIdx.x; kk < k;
         kk += static_cast<int>(gridDim.x) * blockDim.x) {
        const float av = fp8_e4m3fn_to_float(
            a[static_cast<int64_t>(kk) * a_stride1]);
        const float as = load_scale_any(sfa, sfa_type, 0, kk, 1, gran_k_a,
                                        sfa_stride0, sfa_stride1);
        a_dequant[kk] = av * as;
    }
}

template <typename out_t>
__global__ void fp8_c128_m1_predecoded_a_gemm_kernel(
    const float* __restrict__ a_dequant, const uint8_t* __restrict__ b,
    const void* __restrict__ sfb, out_t* __restrict__ d, int sfb_type, int n, int k,
    int64_t b_stride0, int64_t b_stride1, int64_t d_stride1,
    int64_t sfb_stride0, int64_t sfb_stride1, int gran_mn_b, int gran_k_b,
    bool accumulate) {
    __shared__ float reductions[256];
    const int col = blockIdx.x;
    float partial = 0.0f;

    for (int kk = threadIdx.x; kk < k; kk += blockDim.x) {
        const float bv = fp8_e4m3fn_to_float(
            b[static_cast<int64_t>(col) * b_stride0 +
              static_cast<int64_t>(kk) * b_stride1]);
        const float bs = load_scale_any(sfb, sfb_type, col, kk, gran_mn_b,
                                        gran_k_b, sfb_stride0, sfb_stride1);
        partial += a_dequant[kk] * bv * bs;
    }

    reductions[threadIdx.x] = partial;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            reductions[threadIdx.x] += reductions[threadIdx.x + offset];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const int64_t out_offset = static_cast<int64_t>(col) * d_stride1;
        float value = reductions[0];
        if (accumulate)
            value += load_direct_gemm<out_t>(d, out_offset);
        store_direct_gemm<out_t>(d, out_offset, value);
    }
}

template <typename out_t, int kColsPerBlock>
__global__ void fp8_c128_m1_predecoded_a_multi_col_gemm_kernel(
    const float* __restrict__ a_dequant, const uint8_t* __restrict__ b,
    const void* __restrict__ sfb, out_t* __restrict__ d, int sfb_type, int n,
    int k, int64_t b_stride0, int64_t b_stride1, int64_t d_stride1,
    int64_t sfb_stride0, int64_t sfb_stride1, int gran_mn_b, int gran_k_b,
    bool accumulate) {
    __shared__ float reductions[kColsPerBlock][8];
    float partial[kColsPerBlock];
#pragma unroll
    for (int i = 0; i < kColsPerBlock; ++i)
        partial[i] = 0.0f;

    const int col_base = blockIdx.x * kColsPerBlock;
    for (int kk = threadIdx.x; kk < k; kk += blockDim.x) {
        const float av = a_dequant[kk];
#pragma unroll
        for (int i = 0; i < kColsPerBlock; ++i) {
            const int col = col_base + i;
            if (col < n) {
                const float bv = fp8_e4m3fn_to_float(
                    b[static_cast<int64_t>(col) * b_stride0 +
                      static_cast<int64_t>(kk) * b_stride1]);
                const float bs = load_scale_any(
                    sfb, sfb_type, col, kk, gran_mn_b, gran_k_b,
                    sfb_stride0, sfb_stride1);
                partial[i] += av * bv * bs;
            }
        }
    }

    reduce_cols_in_block<kColsPerBlock>(partial, reductions);

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < kColsPerBlock; ++i) {
            const int col = col_base + i;
            if (col < n) {
                const int64_t out_offset =
                    static_cast<int64_t>(col) * d_stride1;
                float value = reductions[i][0];
                if (accumulate)
                    value += load_direct_gemm<out_t>(d, out_offset);
                store_direct_gemm<out_t>(d, out_offset, value);
            }
        }
    }
}

template <typename out_t, int kColsPerBlock>
__global__ void fp8_c128_m1_predecoded_a_kblock_scale_gemm_kernel(
    const float* __restrict__ a_dequant, const uint8_t* __restrict__ b,
    const void* __restrict__ sfb, out_t* __restrict__ d, int sfb_type, int n,
    int k, int64_t b_stride0, int64_t b_stride1, int64_t d_stride1,
    int64_t sfb_stride0, int64_t sfb_stride1, int gran_mn_b, int gran_k_b,
    bool accumulate) {
    __shared__ float reductions[kColsPerBlock][8];
    __shared__ float block_scales[kColsPerBlock];
    float partial[kColsPerBlock];
#pragma unroll
    for (int i = 0; i < kColsPerBlock; ++i)
        partial[i] = 0.0f;

    const int col_base = blockIdx.x * kColsPerBlock;
    for (int k_block_start = 0; k_block_start < k;
         k_block_start += gran_k_b) {
        if (threadIdx.x < kColsPerBlock) {
            const int col = col_base + threadIdx.x;
            block_scales[threadIdx.x] =
                col < n ? load_scale_any(sfb, sfb_type, col, k_block_start,
                                          gran_mn_b, gran_k_b, sfb_stride0,
                                          sfb_stride1)
                        : 0.0f;
        }
        __syncthreads();

        const int k_block_end = min(k, k_block_start + gran_k_b);
        for (int kk = k_block_start + threadIdx.x; kk < k_block_end;
             kk += blockDim.x) {
            const float av = a_dequant[kk];
#pragma unroll
            for (int i = 0; i < kColsPerBlock; ++i) {
                const int col = col_base + i;
                if (col < n) {
                    const float bv = fp8_e4m3fn_to_float(
                        b[static_cast<int64_t>(col) * b_stride0 +
                          static_cast<int64_t>(kk) * b_stride1]);
                    partial[i] += av * bv * block_scales[i];
                }
            }
        }
        __syncthreads();
    }

    reduce_cols_in_block<kColsPerBlock>(partial, reductions);

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < kColsPerBlock; ++i) {
            const int col = col_base + i;
            if (col < n) {
                const int64_t out_offset =
                    static_cast<int64_t>(col) * d_stride1;
                float value = reductions[i][0];
                if (accumulate)
                    value += load_direct_gemm<out_t>(d, out_offset);
                store_direct_gemm<out_t>(d, out_offset, value);
            }
        }
    }
}

template <typename out_t, int kColsPerBlock>
__global__ void fp8_c128_m1_predecoded_a_regscale_gemm_kernel(
    const float* __restrict__ a_dequant, const uint8_t* __restrict__ b,
    const void* __restrict__ sfb, out_t* __restrict__ d, int sfb_type, int n,
    int k, int64_t b_stride0, int64_t b_stride1, int64_t d_stride1,
    int64_t sfb_stride0, int64_t sfb_stride1, int gran_mn_b, int gran_k_b,
    bool accumulate) {
    __shared__ float reductions[kColsPerBlock][8];
    float partial[kColsPerBlock];
#pragma unroll
    for (int i = 0; i < kColsPerBlock; ++i)
        partial[i] = 0.0f;

    const int col_base = blockIdx.x * kColsPerBlock;
    for (int k_block_start = 0; k_block_start < k;
         k_block_start += gran_k_b) {
        float block_scales[kColsPerBlock];
#pragma unroll
        for (int i = 0; i < kColsPerBlock; ++i) {
            const int col = col_base + i;
            block_scales[i] =
                col < n ? load_scale_any(sfb, sfb_type, col, k_block_start,
                                          gran_mn_b, gran_k_b, sfb_stride0,
                                          sfb_stride1)
                        : 0.0f;
        }

        const int k_block_end = min(k, k_block_start + gran_k_b);
        for (int kk = k_block_start + threadIdx.x; kk < k_block_end;
             kk += blockDim.x) {
            const float av = a_dequant[kk];
#pragma unroll
            for (int i = 0; i < kColsPerBlock; ++i) {
                const int col = col_base + i;
                if (col < n) {
                    const float bv = fp8_e4m3fn_to_float(
                        b[static_cast<int64_t>(col) * b_stride0 +
                          static_cast<int64_t>(kk) * b_stride1]);
                    partial[i] += av * bv * block_scales[i];
                }
            }
        }
    }

    reduce_cols_in_block<kColsPerBlock>(partial, reductions);

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < kColsPerBlock; ++i) {
            const int col = col_base + i;
            if (col < n) {
                const int64_t out_offset =
                    static_cast<int64_t>(col) * d_stride1;
                float value = reductions[i][0];
                if (accumulate)
                    value += load_direct_gemm<out_t>(d, out_offset);
                store_direct_gemm<out_t>(d, out_offset, value);
            }
        }
    }
}

template <typename out_t, int kWarpsPerBlock>
__global__ void fp8_c128_m1_predecoded_a_warp_col_gemm_kernel(
    const float* __restrict__ a_dequant, const uint8_t* __restrict__ b,
    const void* __restrict__ sfb, out_t* __restrict__ d, int sfb_type, int n,
    int k, int64_t b_stride0, int64_t b_stride1, int64_t d_stride1,
    int64_t sfb_stride0, int64_t sfb_stride1, int gran_mn_b, int gran_k_b,
    bool accumulate) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= kWarpsPerBlock)
        return;

    const int col = blockIdx.x * kWarpsPerBlock + warp_id;
    if (col >= n)
        return;

    float partial = 0.0f;
    for (int k_block_start = 0; k_block_start < k;
         k_block_start += gran_k_b) {
        float bs = 0.0f;
        if (lane == 0) {
            bs = load_scale_any(sfb, sfb_type, col, k_block_start,
                                gran_mn_b, gran_k_b, sfb_stride0,
                                sfb_stride1);
        }
        bs = __shfl_sync(0xffffffff, bs, 0);

        const int k_block_end = min(k, k_block_start + gran_k_b);
        for (int kk = k_block_start + lane; kk < k_block_end; kk += 32) {
            const float bv = fp8_e4m3fn_to_float(
                b[static_cast<int64_t>(col) * b_stride0 +
                  static_cast<int64_t>(kk) * b_stride1]);
            partial += a_dequant[kk] * bv * bs;
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, offset);

    if (lane == 0) {
        const int64_t out_offset = static_cast<int64_t>(col) * d_stride1;
        float value = partial;
        if (accumulate)
            value += load_direct_gemm<out_t>(d, out_offset);
        store_direct_gemm<out_t>(d, out_offset, value);
    }
}

template <typename out_t, int kDPerBlock>
__global__ void fp8_bhr_hdr_bhd_kernel(
    const uint8_t* __restrict__ a, const void* __restrict__ a_scale,
    const uint8_t* __restrict__ b, const void* __restrict__ b_scale,
    out_t* __restrict__ out, int B, int H, int D, int R, int b_dim,
    int b_scale_dim, int b_scale_d_size, int b_scale_gran_d, int a_scale_type,
    int b_scale_type, int64_t a_s0, int64_t a_s1, int64_t a_s2,
    int64_t b_s0, int64_t b_s1, int64_t b_s2, int64_t out_s0,
    int64_t out_s1, int64_t out_s2, int64_t as_s0, int64_t as_s1,
    int64_t as_s2, int64_t bs_s0, int64_t bs_s1, int64_t bs_s2) {
    __shared__ float reductions[kDPerBlock][8];
    const int linear = blockIdx.x;
    const int d_blocks = (D + kDPerBlock - 1) / kDPerBlock;
    const int d_base = (linear % d_blocks) * kDPerBlock;
    const int h = (linear / d_blocks) % H;
    const int batch = linear / (d_blocks * H);
    float partial[kDPerBlock];
#pragma unroll
    for (int i = 0; i < kDPerBlock; ++i)
        partial[i] = 0.0f;

    const int64_t a_base = static_cast<int64_t>(batch) * a_s0 +
                           static_cast<int64_t>(h) * a_s1;
    const int64_t as_base = static_cast<int64_t>(batch) * as_s0 +
                            static_cast<int64_t>(h) * as_s1;

    for (int rr = threadIdx.x; rr < R; rr += blockDim.x) {
        const int k_block = rr / 128;
        const float av = fp8_e4m3fn_to_float(a[a_base + static_cast<int64_t>(rr) * a_s2]);
        const float as =
            load_vec128_scale(a_scale, a_scale_type, as_base, as_s2, k_block);
        const float a_scaled = av * as;
#pragma unroll
        for (int i = 0; i < kDPerBlock; ++i) {
            const int d = d_base + i;
            if (d < D) {
                const int64_t b_base =
                    b_dim == 3 ? static_cast<int64_t>(h) * b_s0 +
                                     static_cast<int64_t>(d) * b_s1
                               : static_cast<int64_t>(h * D + d) * b_s0;
                const int scale_d = d / b_scale_gran_d;
                const int64_t bs_base =
                    b_scale_dim == 3
                        ? static_cast<int64_t>(h) * bs_s0 +
                              static_cast<int64_t>(scale_d) * bs_s1
                        : static_cast<int64_t>(h * b_scale_d_size + scale_d) *
                              bs_s0;
                const float bv = fp8_e4m3fn_to_float(
                    b[b_base + static_cast<int64_t>(rr) *
                                   (b_dim == 3 ? b_s2 : b_s1)]);
                const float bs = load_vec128_scale(
                    b_scale, b_scale_type, bs_base,
                    b_scale_dim == 3 ? bs_s2 : bs_s1, k_block);
                partial[i] += a_scaled * bv * bs;
            }
        }
    }

    reduce_cols_in_block<kDPerBlock>(partial, reductions);

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < kDPerBlock; ++i) {
            const int d = d_base + i;
            if (d < D) {
                store_direct_gemm<out_t>(
                    out, static_cast<int64_t>(batch) * out_s0 +
                             static_cast<int64_t>(h) * out_s1 +
                             static_cast<int64_t>(d) * out_s2,
                    reductions[i][0]);
            }
        }
    }
}

template <typename out_t, int kDPerBlock>
__global__ void fp8_bhr_hdr_bhd_kblock_scale_kernel(
    const uint8_t* __restrict__ a, const void* __restrict__ a_scale,
    const uint8_t* __restrict__ b, const void* __restrict__ b_scale,
    out_t* __restrict__ out, int B, int H, int D, int R, int b_dim,
    int b_scale_dim, int b_scale_d_size, int b_scale_gran_d, int a_scale_type,
    int b_scale_type, int64_t a_s0, int64_t a_s1, int64_t a_s2,
    int64_t b_s0, int64_t b_s1, int64_t b_s2, int64_t out_s0,
    int64_t out_s1, int64_t out_s2, int64_t as_s0, int64_t as_s1,
    int64_t as_s2, int64_t bs_s0, int64_t bs_s1, int64_t bs_s2) {
    __shared__ float reductions[kDPerBlock][8];
    const int linear = blockIdx.x;
    const int d_blocks = (D + kDPerBlock - 1) / kDPerBlock;
    const int d_base = (linear % d_blocks) * kDPerBlock;
    const int h = (linear / d_blocks) % H;
    const int batch = linear / (d_blocks * H);
    float partial[kDPerBlock];
#pragma unroll
    for (int i = 0; i < kDPerBlock; ++i)
        partial[i] = 0.0f;

    const int64_t a_base = static_cast<int64_t>(batch) * a_s0 +
                           static_cast<int64_t>(h) * a_s1;
    const int64_t as_base = static_cast<int64_t>(batch) * as_s0 +
                            static_cast<int64_t>(h) * as_s1;

    for (int k_block_start = 0, k_block = 0; k_block_start < R;
         k_block_start += 128, ++k_block) {
        const float as =
            load_vec128_scale(a_scale, a_scale_type, as_base, as_s2, k_block);
        float bs_vals[kDPerBlock];
#pragma unroll
        for (int i = 0; i < kDPerBlock; ++i) {
            const int d = d_base + i;
            if (d < D) {
                const int scale_d = d / b_scale_gran_d;
                const int64_t bs_base =
                    b_scale_dim == 3
                        ? static_cast<int64_t>(h) * bs_s0 +
                              static_cast<int64_t>(scale_d) * bs_s1
                        : static_cast<int64_t>(h * b_scale_d_size + scale_d) *
                              bs_s0;
                bs_vals[i] = load_vec128_scale(
                    b_scale, b_scale_type, bs_base,
                    b_scale_dim == 3 ? bs_s2 : bs_s1, k_block);
            } else {
                bs_vals[i] = 0.0f;
            }
        }

        const int k_block_end = min(R, k_block_start + 128);
        for (int rr = k_block_start + threadIdx.x; rr < k_block_end;
             rr += blockDim.x) {
            const float a_scaled =
                fp8_e4m3fn_to_float(
                    a[a_base + static_cast<int64_t>(rr) * a_s2]) *
                as;
#pragma unroll
            for (int i = 0; i < kDPerBlock; ++i) {
                const int d = d_base + i;
                if (d < D) {
                    const int64_t b_base =
                        b_dim == 3 ? static_cast<int64_t>(h) * b_s0 +
                                         static_cast<int64_t>(d) * b_s1
                                   : static_cast<int64_t>(h * D + d) * b_s0;
                    const float bv = fp8_e4m3fn_to_float(
                        b[b_base + static_cast<int64_t>(rr) *
                                       (b_dim == 3 ? b_s2 : b_s1)]);
                    partial[i] += a_scaled * bv * bs_vals[i];
                }
            }
        }
    }

    reduce_cols_in_block<kDPerBlock>(partial, reductions);

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < kDPerBlock; ++i) {
            const int d = d_base + i;
            if (d < D) {
                store_direct_gemm<out_t>(
                    out, static_cast<int64_t>(batch) * out_s0 +
                             static_cast<int64_t>(h) * out_s1 +
                             static_cast<int64_t>(d) * out_s2,
                    reductions[i][0]);
            }
        }
    }
}

__global__ void fp8_bhr_hdr_bhd_m1_mma_kernel(
    const uint8_t* __restrict__ a, const int32_t* __restrict__ a_scale,
    const uint8_t* __restrict__ b, const int32_t* __restrict__ b_scale,
    __nv_bfloat16* __restrict__ out, int B, int H, int D, int R, int b_dim,
    int b_scale_dim, int b_scale_d_size, int b_scale_gran_d, int64_t a_s0,
    int64_t a_s1, int64_t a_s2, int64_t b_s0, int64_t b_s1,
    int64_t b_s2, int64_t out_s0, int64_t out_s1, int64_t out_s2,
    int64_t as_s0, int64_t as_s1, int64_t as_s2, int64_t bs_s0,
    int64_t bs_s1, int64_t bs_s2) {
#if defined(CUTE_ARCH_MXF8F6F4_MMA_ENABLED)
    using MmaOp = cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
        cutlass::float_ue8m0_t, 32>;

    constexpr int kWarpsPerBlock = 4;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int t0 = lane & 3;
    const int t1 = lane >> 2;
    if (warp_id >= kWarpsPerBlock)
        return;

    const int d_tiles_per_h = (D + kWarpsPerBlock * 8 - 1) /
                              (kWarpsPerBlock * 8);
    const int tile = blockIdx.x % d_tiles_per_h;
    const int h = (blockIdx.x / d_tiles_per_h) % H;
    const int batch = blockIdx.x / (d_tiles_per_h * H);
    if (batch >= B)
        return;

    const int col_base = (tile * kWarpsPerBlock + warp_id) * 8;
    if (col_base >= D)
        return;

    const int k32 = R / 32;
    const int gran_k_blocks = 128 / 32;
    const int64_t a_base = static_cast<int64_t>(batch) * a_s0 +
                           static_cast<int64_t>(h) * a_s1;
    const int64_t as_base = static_cast<int64_t>(batch) * as_s0 +
                            static_cast<int64_t>(h) * as_s1;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll 1
    for (int kb = 0; kb < k32; ++kb) {
        uint32_t a_reg[4] = {0u, 0u, 0u, 0u};
        uint32_t b_reg[2] = {0u, 0u};

#pragma unroll
        for (int v = 0; v < 16; ++v) {
            const int v0 = v & 3;
            const int v1 = (v >> 2) & 1;
            const int v2 = v >> 3;
            const int offset = t0 * 64 + t1 + v0 * 16 + v1 * 8 + v2 * 256;
            const int local_row = offset & 15;
            const int kk = offset >> 4;
            uint8_t value = 0;
            if (local_row == 0) {
                value = a[a_base + static_cast<int64_t>(kb * 32 + kk) * a_s2];
            }
            a_reg[v >> 2] |= static_cast<uint32_t>(value) << ((v & 3) * 8);
        }

#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int v0 = v & 3;
            const int v1 = v >> 2;
            const int offset = t0 * 32 + t1 + v0 * 8 + v1 * 128;
            const int kk = offset >> 3;
            const int n_local = offset & 7;
            const int d = col_base + n_local;
            uint8_t value = 0;
            if (d < D) {
                const int64_t b_base =
                    b_dim == 3 ? static_cast<int64_t>(h) * b_s0 +
                                     static_cast<int64_t>(d) * b_s1
                               : static_cast<int64_t>(h * D + d) * b_s0;
                value = b[b_base + static_cast<int64_t>(kb * 32 + kk) *
                                       (b_dim == 3 ? b_s2 : b_s1)];
            }
            b_reg[v >> 2] |= static_cast<uint32_t>(value) << ((v & 3) * 8);
        }

        const int k_block = kb / gran_k_blocks;
        const uint8_t sfa_value =
            load_ue8m0_raw_packed(a_scale, as_base, as_s2, k_block);
        const int scale_d = min(col_base + t1, D - 1) / b_scale_gran_d;
        const int64_t bs_base =
            b_scale_dim == 3
                ? static_cast<int64_t>(h) * bs_s0 +
                      static_cast<int64_t>(scale_d) * bs_s1
                : static_cast<int64_t>(h * b_scale_d_size + scale_d) * bs_s0;
        const uint8_t sfb_value =
            load_ue8m0_raw_packed(b_scale, bs_base, bs_s2, k_block);

        MmaOp::fma(acc[0], acc[1], acc[2], acc[3], a_reg[0], a_reg[1],
                   a_reg[2], a_reg[3], b_reg[0], b_reg[1], acc[0], acc[1],
                   acc[2], acc[3], sfa_value, sfb_value);
    }

    if (t1 == 0) {
        const int d0 = col_base + t0 * 2;
        const int64_t out_base = static_cast<int64_t>(batch) * out_s0 +
                                 static_cast<int64_t>(h) * out_s1;
        if (d0 < D)
            out[out_base + static_cast<int64_t>(d0) * out_s2] =
                __float2bfloat16(acc[0]);
        const int d1 = d0 + 1;
        if (d1 < D)
            out[out_base + static_cast<int64_t>(d1) * out_s2] =
                __float2bfloat16(acc[1]);
    }
#endif
}

template <typename out_t, int kWarpsPerBlock>
__global__ void fp8_bhr_hdr_bhd_warp_single_col_kernel(
    const uint8_t* __restrict__ a, const void* __restrict__ a_scale,
    const uint8_t* __restrict__ b, const void* __restrict__ b_scale,
    out_t* __restrict__ out, int B, int H, int D, int R, int b_dim,
    int b_scale_dim, int b_scale_d_size, int b_scale_gran_d, int a_scale_type,
    int b_scale_type, int64_t a_s0, int64_t a_s1, int64_t a_s2,
    int64_t b_s0, int64_t b_s1, int64_t b_s2, int64_t out_s0,
    int64_t out_s1, int64_t out_s2, int64_t as_s0, int64_t as_s1,
    int64_t as_s2, int64_t bs_s0, int64_t bs_s1, int64_t bs_s2) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= kWarpsPerBlock)
        return;

    const int linear_col = blockIdx.x * kWarpsPerBlock + warp_id;
    const int d = linear_col % D;
    const int h = (linear_col / D) % H;
    const int batch = linear_col / (D * H);
    if (batch >= B)
        return;

    const int64_t a_base = static_cast<int64_t>(batch) * a_s0 +
                           static_cast<int64_t>(h) * a_s1;
    const int64_t as_base = static_cast<int64_t>(batch) * as_s0 +
                            static_cast<int64_t>(h) * as_s1;
    const int64_t b_base =
        b_dim == 3 ? static_cast<int64_t>(h) * b_s0 +
                         static_cast<int64_t>(d) * b_s1
                   : static_cast<int64_t>(h * D + d) * b_s0;
    const int scale_d = d / b_scale_gran_d;
    const int64_t bs_base =
        b_scale_dim == 3
            ? static_cast<int64_t>(h) * bs_s0 +
                  static_cast<int64_t>(scale_d) * bs_s1
            : static_cast<int64_t>(h * b_scale_d_size + scale_d) * bs_s0;
    const int64_t b_r_stride = b_dim == 3 ? b_s2 : b_s1;
    const int64_t bs_k_stride = b_scale_dim == 3 ? bs_s2 : bs_s1;

    float partial = 0.0f;
    for (int k_block_start = 0, k_block = 0; k_block_start < R;
         k_block_start += 128, ++k_block) {
        const float as =
            load_vec128_scale(a_scale, a_scale_type, as_base, as_s2, k_block);
        const float bs = load_vec128_scale(b_scale, b_scale_type, bs_base,
                                           bs_k_stride, k_block);
        const float scale = as * bs;
        const int k_block_end = min(R, k_block_start + 128);
        for (int rr = k_block_start + lane; rr < k_block_end; rr += 32) {
            const float av =
                fp8_e4m3fn_to_float(a[a_base + static_cast<int64_t>(rr) * a_s2]);
            const float bv =
                fp8_e4m3fn_to_float(b[b_base + static_cast<int64_t>(rr) *
                                                   b_r_stride]);
            partial += av * bv * scale;
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(0xffffffff, partial, offset);

    if (lane == 0) {
        store_direct_gemm<out_t>(
            out, static_cast<int64_t>(batch) * out_s0 +
                     static_cast<int64_t>(h) * out_s1 +
                     static_cast<int64_t>(d) * out_s2,
            partial);
    }
}

template <typename out_t, int kWarpsPerBlock, int kColsPerWarp>
__global__ void fp8_bhr_hdr_bhd_warp_multi_col_kernel(
    const uint8_t* __restrict__ a, const void* __restrict__ a_scale,
    const uint8_t* __restrict__ b, const void* __restrict__ b_scale,
    out_t* __restrict__ out, int B, int H, int D, int R, int b_dim,
    int b_scale_dim, int b_scale_d_size, int b_scale_gran_d, int a_scale_type,
    int b_scale_type, int64_t a_s0, int64_t a_s1, int64_t a_s2,
    int64_t b_s0, int64_t b_s1, int64_t b_s2, int64_t out_s0,
    int64_t out_s1, int64_t out_s2, int64_t as_s0, int64_t as_s1,
    int64_t as_s2, int64_t bs_s0, int64_t bs_s1, int64_t bs_s2) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= kWarpsPerBlock)
        return;

    const int d_groups = (D + kColsPerWarp - 1) / kColsPerWarp;
    const int linear_group = blockIdx.x * kWarpsPerBlock + warp_id;
    const int d_base = (linear_group % d_groups) * kColsPerWarp;
    const int h = (linear_group / d_groups) % H;
    const int batch = linear_group / (d_groups * H);
    if (batch >= B)
        return;

    const int64_t a_base = static_cast<int64_t>(batch) * a_s0 +
                           static_cast<int64_t>(h) * a_s1;
    const int64_t as_base = static_cast<int64_t>(batch) * as_s0 +
                            static_cast<int64_t>(h) * as_s1;
    const int64_t b_r_stride = b_dim == 3 ? b_s2 : b_s1;
    const int64_t bs_k_stride = b_scale_dim == 3 ? bs_s2 : bs_s1;

    float partial[kColsPerWarp];
#pragma unroll
    for (int i = 0; i < kColsPerWarp; ++i)
        partial[i] = 0.0f;
    for (int k_block_start = 0, k_block = 0; k_block_start < R;
         k_block_start += 128, ++k_block) {
        const float as =
            load_vec128_scale(a_scale, a_scale_type, as_base, as_s2, k_block);
        float scale[kColsPerWarp];
        int64_t b_base[kColsPerWarp];
#pragma unroll
        for (int i = 0; i < kColsPerWarp; ++i) {
            const int d = d_base + i;
            if (d < D) {
                b_base[i] = b_dim == 3
                                ? static_cast<int64_t>(h) * b_s0 +
                                      static_cast<int64_t>(d) * b_s1
                                : static_cast<int64_t>(h * D + d) * b_s0;
                const int scale_d = d / b_scale_gran_d;
                const int64_t bs_base =
                    b_scale_dim == 3
                        ? static_cast<int64_t>(h) * bs_s0 +
                              static_cast<int64_t>(scale_d) * bs_s1
                        : static_cast<int64_t>(h * b_scale_d_size + scale_d) *
                              bs_s0;
                const float bs = load_vec128_scale(
                    b_scale, b_scale_type, bs_base, bs_k_stride, k_block);
                scale[i] = as * bs;
            } else {
                b_base[i] = 0;
                scale[i] = 0.0f;
            }
        }
        const int k_block_end = min(R, k_block_start + 128);
        for (int rr = k_block_start + lane; rr < k_block_end; rr += 32) {
            const float av =
                fp8_e4m3fn_to_float(a[a_base + static_cast<int64_t>(rr) * a_s2]);
#pragma unroll
            for (int i = 0; i < kColsPerWarp; ++i) {
                const int d = d_base + i;
                if (d < D) {
                    const float bv = fp8_e4m3fn_to_float(
                        b[b_base[i] + static_cast<int64_t>(rr) * b_r_stride]);
                    partial[i] += av * bv * scale[i];
                }
            }
        }
    }

#pragma unroll
    for (int i = 0; i < kColsPerWarp; ++i) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            partial[i] += __shfl_down_sync(0xffffffff, partial[i], offset);
    }

    if (lane == 0) {
        const int64_t out_base = static_cast<int64_t>(batch) * out_s0 +
                                 static_cast<int64_t>(h) * out_s1;
#pragma unroll
        for (int i = 0; i < kColsPerWarp; ++i) {
            const int d = d_base + i;
            if (d < D) {
                store_direct_gemm<out_t>(
                    out, out_base + static_cast<int64_t>(d) * out_s2,
                    partial[i]);
            }
        }
    }
}

__device__ __forceinline__ int sm120_group_for_row_contiguous(
    const int32_t* grouped_layout, int row) {
    return grouped_layout[row];
}

__device__ __forceinline__ int sm120_group_for_row_psum(
    const int32_t* grouped_layout, int num_groups, int row) {
    int aligned_start = 0;
    for (int group = 0; group < num_groups; ++group) {
        const int actual_end = grouped_layout[group];
        const int aligned_end = ((actual_end + 127) / 128) * 128;
        if (row >= aligned_start && row < actual_end)
            return group;
        if (row >= actual_end && row < aligned_end)
            return -1;
        aligned_start = aligned_end;
    }
    return -1;
}

template <typename out_t, bool kUsePsumLayout, int kColsPerBlock>
__global__ void grouped_fp8_fp4_contiguous_kernel(
    const uint8_t* __restrict__ a, const float* __restrict__ sfa,
    const int8_t* __restrict__ b, const float* __restrict__ sfb,
    out_t* __restrict__ d, const int32_t* __restrict__ grouped_layout,
    int num_groups, int m, int n, int k, int64_t a_stride0,
    int64_t a_stride1, int64_t b_stride0, int64_t b_stride1,
    int64_t b_stride2, int64_t d_stride0, int64_t d_stride1,
    int64_t sfa_stride0, int64_t sfa_stride1, int64_t sfb_stride0,
    int64_t sfb_stride1, int64_t sfb_stride2, int gran_k_a, int gran_k_b,
    bool b_k_major, bool zero_padded_rows) {
    __shared__ float reductions[kColsPerBlock][8];

    const int linear = blockIdx.x;
    const int col_blocks = (n + kColsPerBlock - 1) / kColsPerBlock;
    const int row = linear / col_blocks;
    const int col_base = (linear - row * col_blocks) * kColsPerBlock;
    const int group = kUsePsumLayout
                          ? sm120_group_for_row_psum(grouped_layout, num_groups, row)
                          : sm120_group_for_row_contiguous(grouped_layout, row);

    if (group < 0) {
        if (!zero_padded_rows)
            return;
        if (threadIdx.x == 0) {
#pragma unroll
            for (int i = 0; i < kColsPerBlock; ++i) {
                const int col = col_base + i;
                if (col < n) {
                    store_direct_gemm<out_t>(
                        d,
                        static_cast<int64_t>(row) * d_stride0 +
                            static_cast<int64_t>(col) * d_stride1,
                        0.0f);
                }
            }
        }
        return;
    }

    float partial[kColsPerBlock];
#pragma unroll
    for (int i = 0; i < kColsPerBlock; ++i)
        partial[i] = 0.0f;

    const int packed_k_count = k >> 1;
    for (int packed_k = threadIdx.x; packed_k < packed_k_count;
         packed_k += blockDim.x) {
        const int kk0 = packed_k << 1;
        const int kk1 = kk0 + 1;
        const float as = sfa[static_cast<int64_t>(row) * sfa_stride0 +
                             static_cast<int64_t>(kk0 / gran_k_a) *
                                 sfa_stride1];
        const float a0 = fp8_e4m3fn_to_float(
                             a[static_cast<int64_t>(row) * a_stride0 +
                               static_cast<int64_t>(kk0) * a_stride1]) *
                         as;
        const float a1 = fp8_e4m3fn_to_float(
                             a[static_cast<int64_t>(row) * a_stride0 +
                               static_cast<int64_t>(kk1) * a_stride1]) *
                         as;
#pragma unroll
        for (int i = 0; i < kColsPerBlock; ++i) {
            const int col = col_base + i;
            if (col < n) {
                const uint8_t packed = static_cast<uint8_t>(
                    b_k_major
                        ? b[static_cast<int64_t>(group) * b_stride0 +
                            static_cast<int64_t>(col) * b_stride1 +
                            packed_k * b_stride2]
                        : b[static_cast<int64_t>(group) * b_stride0 +
                            static_cast<int64_t>(packed_k) * b_stride1 +
                            static_cast<int64_t>(col / 2) * b_stride2]);
                const float b0 = fp4_e2m1_to_float(packed & 0x0f);
                const float b1 = fp4_e2m1_to_float((packed >> 4) & 0x0f);
                const float bs =
                    sfb[static_cast<int64_t>(group) * sfb_stride0 +
                        static_cast<int64_t>(col) * sfb_stride1 +
                        static_cast<int64_t>(kk0 / gran_k_b) * sfb_stride2];
                partial[i] += (a0 * b0 + a1 * b1) * bs;
            }
        }
    }

    reduce_cols_in_block<kColsPerBlock>(partial, reductions);

    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < kColsPerBlock; ++i) {
            const int col = col_base + i;
            if (col < n) {
                store_direct_gemm<out_t>(
                    d,
                    static_cast<int64_t>(row) * d_stride0 +
                        static_cast<int64_t>(col) * d_stride1,
                    reductions[i][0]);
            }
        }
    }
}

int fallback_grid(int64_t total) {
    constexpr int threads = 256;
    const int64_t blocks = (total + threads - 1) / threads;
    return static_cast<int>(std::min<int64_t>(blocks, 4096));
}

int scale_tensor_type(const torch::Tensor& scale) {
    if (scale.scalar_type() == torch::kFloat32)
        return 0;
    if (scale.scalar_type() == torch::kInt32)
        return 1;
    DG_HOST_UNREACHABLE("SM120 FP8 C128 scales must be float32 or packed int32 UE8M0");
}

void check_c128_scale_shape(const torch::Tensor& scale, int64_t rows,
                            int64_t cols, int gran_mn, int gran_k) {
    const int scale_type = scale_tensor_type(scale);
    DG_HOST_ASSERT(scale.dim() == 2);
    DG_HOST_ASSERT(scale.size(0) == (rows + gran_mn - 1) / gran_mn);
    const int64_t k_blocks = (cols + gran_k - 1) / gran_k;
    const int64_t expected_cols =
        scale_type == 0 ? k_blocks : (k_blocks + 3) / 4;
    DG_HOST_ASSERT(scale.size(1) == expected_cols);
}

torch::Tensor dequant_fp8_c128_to_bf16(const torch::Tensor& x,
                                       const torch::Tensor& scale,
                                       int64_t rows, int64_t cols,
                                       int gran_mn, int gran_k) {
    DG_HOST_ASSERT(x.scalar_type() == torch::kFloat8_e4m3fn);
    check_c128_scale_shape(scale, rows, cols, gran_mn, gran_k);
    const int scale_type = scale_tensor_type(scale);

    auto out = torch::empty({rows, cols}, x.options().dtype(torch::kBFloat16));
    constexpr int threads = 256;
    const int64_t total = rows * cols;
    const int grid = fallback_grid(total);
    const auto stream = at::cuda::getCurrentCUDAStream();

    dequant_fp8_c128_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(x.data_ptr()),
        scale.data_ptr(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        scale_type,
        rows, cols,
        x.stride(0), x.stride(1),
        scale.stride(0), scale.stride(1),
        gran_mn, gran_k);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
    return out;
}

void direct_fp8_c128_gemm(const torch::Tensor& a, const torch::Tensor& sfa,
                          const torch::Tensor& b, const torch::Tensor& sfb,
                          const torch::Tensor& d, int m, int n, int k,
                          int gran_mn_a, int gran_k_a, int gran_mn_b,
                          int gran_k_b, bool accumulate) {
    constexpr int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    constexpr int cols_per_block = 16;
    const int grid = m * ((n + cols_per_block - 1) / cols_per_block);
    const int sfa_type = scale_tensor_type(sfa);
    const int sfb_type = scale_tensor_type(sfb);
    if (d.scalar_type() == torch::kBFloat16) {
        fp8_c128_direct_gemm_kernel<__nv_bfloat16>
            <<<grid, threads, 0, stream>>>(
                reinterpret_cast<const uint8_t*>(a.data_ptr()),
                sfa.data_ptr(),
                reinterpret_cast<const uint8_t*>(b.data_ptr()),
                sfb.data_ptr(),
                reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), sfa_type, sfb_type, m, n, k,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                d.stride(0), d.stride(1), sfa.stride(0), sfa.stride(1),
                sfb.stride(0), sfb.stride(1), gran_mn_a, gran_k_a,
                gran_mn_b, gran_k_b, accumulate);
    } else if (d.scalar_type() == torch::kFloat32) {
        fp8_c128_direct_gemm_kernel<float><<<grid, threads, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(a.data_ptr()), sfa.data_ptr(),
            reinterpret_cast<const uint8_t*>(b.data_ptr()), sfb.data_ptr(),
            d.data_ptr<float>(), sfa_type, sfb_type, m, n, k, a.stride(0), a.stride(1),
            b.stride(0), b.stride(1), d.stride(0), d.stride(1),
            sfa.stride(0), sfa.stride(1), sfb.stride(0), sfb.stride(1),
            gran_mn_a, gran_k_a, gran_mn_b, gran_k_b, accumulate);
    } else {
        DG_HOST_UNREACHABLE("Unsupported output dtype for SM120 direct FP8 GEMM");
    }
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

void predecoded_a_fp8_c128_gemm_m1(const torch::Tensor& a,
                                   const torch::Tensor& sfa,
                                   const torch::Tensor& b,
                                   const torch::Tensor& sfb,
                                   const torch::Tensor& d, int n, int k,
                                   int gran_k_a, int gran_mn_b,
                                   int gran_k_b, bool accumulate) {
    auto a_dequant = torch::empty({k}, a.options().dtype(torch::kFloat32));
    constexpr int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const int dequant_grid = fallback_grid(k);
    const int sfa_type = scale_tensor_type(sfa);
    const int sfb_type = scale_tensor_type(sfb);
    fp8_c128_dequant_a_m1_kernel<<<dequant_grid, threads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(a.data_ptr()), sfa.data_ptr(),
        a_dequant.data_ptr<float>(), sfa_type, k, a.stride(1), sfa.stride(0),
        sfa.stride(1), gran_k_a);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    const bool use_kblock_scale =
        std::getenv("DG_SM120_ENABLE_FP8_M1_KBLOCK") != nullptr &&
        gran_k_b > 0 && gran_k_b <= 256;
    const bool use_regscale =
        std::getenv("DG_SM120_ENABLE_FP8_M1_REGSCALE") != nullptr &&
        gran_k_b > 0 && gran_k_b <= 256;
    const bool use_warp_col =
        (std::getenv("DG_SM120_ENABLE_FP8_M1_WARPCOL") != nullptr ||
         (std::getenv("DG_SM120_ENABLE_FP8_M1_WARPCOL_HEURISTIC") != nullptr &&
          n <= 2048 && k >= 4096)) &&
        gran_k_b > 0 && gran_k_b <= 256;
    const char* kblock_cols_env = std::getenv("DG_SM120_FP8_M1_KBLOCK_COLS");
    const int kblock_cols =
        kblock_cols_env != nullptr
            ? env_int_or_default("DG_SM120_FP8_M1_KBLOCK_COLS", 4)
            : ((k <= 2048 && n >= 8192) ? 8 : 4);
    const int kblock_threads =
        env_int_or_default("DG_SM120_FP8_M1_KBLOCK_THREADS", 128);

    if (d.scalar_type() == torch::kBFloat16) {
        if (use_warp_col && kblock_cols >= 8) {
            fp8_c128_m1_predecoded_a_warp_col_gemm_kernel<__nv_bfloat16, 8>
                <<<(n + 7) / 8, 8 * 32, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(),
                    reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), sfb_type,
                    n, k, b.stride(0), b.stride(1), d.stride(1),
                    sfb.stride(0), sfb.stride(1), gran_mn_b, gran_k_b,
                    accumulate);
        } else if (use_warp_col) {
            fp8_c128_m1_predecoded_a_warp_col_gemm_kernel<__nv_bfloat16, 4>
                <<<(n + 3) / 4, 4 * 32, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(),
                    reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), sfb_type,
                    n, k, b.stride(0), b.stride(1), d.stride(1),
                    sfb.stride(0), sfb.stride(1), gran_mn_b, gran_k_b,
                    accumulate);
        } else if (use_regscale && kblock_cols >= 16) {
            fp8_c128_m1_predecoded_a_regscale_gemm_kernel<__nv_bfloat16, 16>
                <<<(n + 15) / 16, kblock_threads, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(),
                    reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), sfb_type,
                    n, k, b.stride(0), b.stride(1), d.stride(1),
                    sfb.stride(0), sfb.stride(1), gran_mn_b, gran_k_b,
                    accumulate);
        } else if (use_regscale && kblock_cols >= 8) {
            fp8_c128_m1_predecoded_a_regscale_gemm_kernel<__nv_bfloat16, 8>
                <<<(n + 7) / 8, kblock_threads, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(),
                    reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), sfb_type,
                    n, k, b.stride(0), b.stride(1), d.stride(1),
                    sfb.stride(0), sfb.stride(1), gran_mn_b, gran_k_b,
                    accumulate);
        } else if (use_regscale) {
            fp8_c128_m1_predecoded_a_regscale_gemm_kernel<__nv_bfloat16, 4>
                <<<(n + 3) / 4, kblock_threads, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(),
                    reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), sfb_type,
                    n, k, b.stride(0), b.stride(1), d.stride(1),
                    sfb.stride(0), sfb.stride(1), gran_mn_b, gran_k_b,
                    accumulate);
        } else if (use_kblock_scale && kblock_cols >= 16) {
            fp8_c128_m1_predecoded_a_kblock_scale_gemm_kernel<__nv_bfloat16, 16>
                <<<(n + 15) / 16, kblock_threads, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(),
                    reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), sfb_type,
                    n, k, b.stride(0), b.stride(1), d.stride(1),
                    sfb.stride(0), sfb.stride(1), gran_mn_b, gran_k_b,
                    accumulate);
        } else if (use_kblock_scale && kblock_cols >= 8) {
            fp8_c128_m1_predecoded_a_kblock_scale_gemm_kernel<__nv_bfloat16, 8>
                <<<(n + 7) / 8, kblock_threads, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(),
                    reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), sfb_type,
                    n, k, b.stride(0), b.stride(1), d.stride(1),
                    sfb.stride(0), sfb.stride(1), gran_mn_b, gran_k_b,
                    accumulate);
        } else if (use_kblock_scale) {
            fp8_c128_m1_predecoded_a_kblock_scale_gemm_kernel<__nv_bfloat16, 4>
                <<<(n + 3) / 4, kblock_threads, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(),
                    reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), sfb_type,
                    n, k, b.stride(0), b.stride(1), d.stride(1),
                    sfb.stride(0), sfb.stride(1), gran_mn_b, gran_k_b,
                    accumulate);
        } else if (std::getenv("DG_SM120_DISABLE_M1_MULTICOL") == nullptr) {
            fp8_c128_m1_predecoded_a_multi_col_gemm_kernel<__nv_bfloat16, 4>
                <<<(n + 3) / 4, threads, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(),
                    reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), sfb_type,
                    n, k, b.stride(0), b.stride(1), d.stride(1),
                    sfb.stride(0), sfb.stride(1), gran_mn_b, gran_k_b,
                    accumulate);
        } else {
            fp8_c128_m1_predecoded_a_gemm_kernel<__nv_bfloat16>
                <<<n, threads, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(),
                    reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), sfb_type,
                    n, k, b.stride(0), b.stride(1), d.stride(1),
                    sfb.stride(0), sfb.stride(1), gran_mn_b, gran_k_b,
                    accumulate);
        }
    } else if (d.scalar_type() == torch::kFloat32) {
        if (use_warp_col && kblock_cols >= 8) {
            fp8_c128_m1_predecoded_a_warp_col_gemm_kernel<float, 8>
                <<<(n + 7) / 8, 8 * 32, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(), d.data_ptr<float>(), sfb_type, n, k,
                    b.stride(0), b.stride(1), d.stride(1), sfb.stride(0),
                    sfb.stride(1), gran_mn_b, gran_k_b, accumulate);
        } else if (use_warp_col) {
            fp8_c128_m1_predecoded_a_warp_col_gemm_kernel<float, 4>
                <<<(n + 3) / 4, 4 * 32, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(), d.data_ptr<float>(), sfb_type, n, k,
                    b.stride(0), b.stride(1), d.stride(1), sfb.stride(0),
                    sfb.stride(1), gran_mn_b, gran_k_b, accumulate);
        } else if (use_regscale && kblock_cols >= 16) {
            fp8_c128_m1_predecoded_a_regscale_gemm_kernel<float, 16>
                <<<(n + 15) / 16, kblock_threads, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(), d.data_ptr<float>(), sfb_type, n, k,
                    b.stride(0), b.stride(1), d.stride(1), sfb.stride(0),
                    sfb.stride(1), gran_mn_b, gran_k_b, accumulate);
        } else if (use_regscale && kblock_cols >= 8) {
            fp8_c128_m1_predecoded_a_regscale_gemm_kernel<float, 8>
                <<<(n + 7) / 8, kblock_threads, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(), d.data_ptr<float>(), sfb_type, n, k,
                    b.stride(0), b.stride(1), d.stride(1), sfb.stride(0),
                    sfb.stride(1), gran_mn_b, gran_k_b, accumulate);
        } else if (use_regscale) {
            fp8_c128_m1_predecoded_a_regscale_gemm_kernel<float, 4>
                <<<(n + 3) / 4, kblock_threads, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(), d.data_ptr<float>(), sfb_type, n, k,
                    b.stride(0), b.stride(1), d.stride(1), sfb.stride(0),
                    sfb.stride(1), gran_mn_b, gran_k_b, accumulate);
        } else if (use_kblock_scale && kblock_cols >= 16) {
            fp8_c128_m1_predecoded_a_kblock_scale_gemm_kernel<float, 16>
                <<<(n + 15) / 16, kblock_threads, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(), d.data_ptr<float>(), sfb_type, n, k,
                    b.stride(0), b.stride(1), d.stride(1), sfb.stride(0),
                    sfb.stride(1), gran_mn_b, gran_k_b, accumulate);
        } else if (use_kblock_scale && kblock_cols >= 8) {
            fp8_c128_m1_predecoded_a_kblock_scale_gemm_kernel<float, 8>
                <<<(n + 7) / 8, kblock_threads, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(), d.data_ptr<float>(), sfb_type, n, k,
                    b.stride(0), b.stride(1), d.stride(1), sfb.stride(0),
                    sfb.stride(1), gran_mn_b, gran_k_b, accumulate);
        } else if (use_kblock_scale) {
            fp8_c128_m1_predecoded_a_kblock_scale_gemm_kernel<float, 4>
                <<<(n + 3) / 4, kblock_threads, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(), d.data_ptr<float>(), sfb_type, n, k,
                    b.stride(0), b.stride(1), d.stride(1), sfb.stride(0),
                    sfb.stride(1), gran_mn_b, gran_k_b, accumulate);
        } else if (std::getenv("DG_SM120_DISABLE_M1_MULTICOL") == nullptr) {
            fp8_c128_m1_predecoded_a_multi_col_gemm_kernel<float, 4>
                <<<(n + 3) / 4, threads, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(), d.data_ptr<float>(), sfb_type, n, k,
                    b.stride(0), b.stride(1), d.stride(1), sfb.stride(0),
                    sfb.stride(1), gran_mn_b, gran_k_b, accumulate);
        } else {
            fp8_c128_m1_predecoded_a_gemm_kernel<float>
                <<<n, threads, 0, stream>>>(
                    a_dequant.data_ptr<float>(),
                    reinterpret_cast<const uint8_t*>(b.data_ptr()),
                    sfb.data_ptr(), d.data_ptr<float>(), sfb_type, n, k,
                    b.stride(0), b.stride(1), d.stride(1), sfb.stride(0),
                    sfb.stride(1), gran_mn_b, gran_k_b, accumulate);
        }
    } else {
        DG_HOST_UNREACHABLE("Unsupported output dtype for SM120 FP8 GEMM");
    }
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

bool try_cublaslt_fp8_c128_gemm(const torch::Tensor& a, const torch::Tensor& sfa,
                                const torch::Tensor& b,
                                const torch::Tensor& sfb,
                                const torch::Tensor& d, int m, int n, int k,
                                int gran_mn_a, int gran_mn_b,
                                const cute::UMMA::Major& major_a,
                                const cute::UMMA::Major& major_b,
                                bool accumulate) {
    if (major_a != cute::UMMA::Major::K ||
        major_b != cute::UMMA::Major::K ||
        gran_mn_a != 1 || gran_mn_b != 128 ||
        sfa.scalar_type() != torch::kFloat32 ||
        sfb.scalar_type() != torch::kFloat32 ||
        d.scalar_type() != torch::kBFloat16)
        return false;

    const auto trans_a = CUBLAS_OP_T;
    const auto trans_b = CUBLAS_OP_N;
    const auto cuda_type_a = at::cuda::ScalarTypeToCudaDataType(b.scalar_type());
    const auto cuda_type_b = at::cuda::ScalarTypeToCudaDataType(a.scalar_type());
    const auto cuda_type_d = at::cuda::ScalarTypeToCudaDataType(d.scalar_type());

    cublasLtMatrixLayout_t layout_a = nullptr;
    cublasLtMatrixLayout_t layout_b = nullptr;
    cublasLtMatrixLayout_t layout_d = nullptr;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    auto cleanup = [&]() {
        destroy_if(pref);
        destroy_if(desc);
        destroy_if(layout_a);
        destroy_if(layout_b);
        destroy_if(layout_d);
    };

    cublasStatus_t status =
        cublasLtMatrixLayoutCreate(&layout_a, cuda_type_a, k, n, b.stride(0));
    if (status != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }
    status = cublasLtMatrixLayoutCreate(&layout_b, cuda_type_b, k, m,
                                        a.stride(0));
    if (status != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }
    status = cublasLtMatrixLayoutCreate(&layout_d, cuda_type_d, n, m,
                                        d.stride(0));
    if (status != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }

    const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    const cudaDataType_t scale_type = CUDA_R_32F;
    status = cublasLtMatmulDescCreate(&desc, compute_type, scale_type);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }
    status = cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a));
    if (status != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }
    status = cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b));
    if (status != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }
    status = cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_SCALE_TYPE, &scale_type,
        sizeof(scale_type));
    if (status != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }

    const cublasLtMatmulMatrixScale_t scale_mode_a =
        CUBLASLT_MATMUL_MATRIX_SCALE_BLK128x128_32F;
    const cublasLtMatmulMatrixScale_t scale_mode_b =
        CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_32F;
    const void* scale_a = sfb.data_ptr<float>();
    const void* scale_b = sfa.data_ptr<float>();
    status = cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a,
        sizeof(scale_a));
    if (status != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }
    status = cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b,
        sizeof(scale_b));
    if (status != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }
    status = cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode_a,
        sizeof(scale_mode_a));
    if (status != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }
    status = cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode_b,
        sizeof(scale_mode_b));
    if (status != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }

#if DG_CUBLASLT_ADVANCED_FEATURES_COMPATIBLE
    const int math_sms = device_runtime->get_num_sms();
    status = cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &math_sms,
        sizeof(math_sms));
    if (status != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }
#endif

    bool fp8_fast_accumulate = false;
    status = cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fp8_fast_accumulate,
        sizeof(fp8_fast_accumulate));
    if (status != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }

    const auto handle = device_runtime->get_cublaslt_handle();
    const auto workspace = device_runtime->get_cublaslt_workspace();
    const auto workspace_bytes = workspace.nbytes();
    const auto stream = at::cuda::getCurrentCUDAStream();

    status = cublasLtMatmulPreferenceCreate(&pref);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }
    status = cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes,
        sizeof(workspace_bytes));
    if (status != CUBLAS_STATUS_SUCCESS) {
        cleanup();
        return false;
    }

    cublasLtMatmulHeuristicResult_t heuristic{};
    int num_heuristic_results = 0;
    status = cublasLtMatmulAlgoGetHeuristic(
        handle, desc, layout_a, layout_b, layout_d, layout_d, pref, 1,
        &heuristic, &num_heuristic_results);
    if (status != CUBLAS_STATUS_SUCCESS || num_heuristic_results != 1) {
        cleanup();
        return false;
    }

    const float alpha = 1.0f;
    const float beta = accumulate ? 1.0f : 0.0f;
    status = cublasLtMatmul(
        handle, desc, &alpha, b.data_ptr(), layout_a, a.data_ptr(), layout_b,
        &beta, d.data_ptr(), layout_d, d.data_ptr(), layout_d,
        &heuristic.algo, workspace.data_ptr(), workspace_bytes, stream);
    cleanup();
    return status == CUBLAS_STATUS_SUCCESS;
}

} // namespace

void sm120_fp8_gemm_nt_fallback(const torch::Tensor& a, const torch::Tensor& sfa,
                                const torch::Tensor& b, const torch::Tensor& sfb,
                                const std::optional<torch::Tensor>& c,
                                const torch::Tensor& d,
                                int m, int n, int k,
                                int gran_mn_a, int gran_k_a,
                                int gran_mn_b, int gran_k_b,
                                const cute::UMMA::Major& major_a,
                                const cute::UMMA::Major& major_b) {
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kFloat32 or sfa.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(sfb.scalar_type() == torch::kFloat32 or sfb.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);

    const auto stream = at::cuda::getCurrentCUDAStream();
    static sm120_profile::KernelProfileCounter profile_counter(
        "sm120_fp8_gemm_fallback");
    sm120_profile::ScopedTimer profile_timer(
        profile_counter, stream, m, n, k, 1);

    if (m == 1 and major_a == cute::UMMA::Major::K and
        major_b == cute::UMMA::Major::K) {
        if (std::getenv("DG_DISABLE_PREDECODED_A_M1") == nullptr) {
            predecoded_a_fp8_c128_gemm_m1(a, sfa, b, sfb, d, n, k,
                                          gran_k_a, gran_mn_b, gran_k_b,
                                          c.has_value());
            return;
        }
        direct_fp8_c128_gemm(a, sfa, b, sfb, d, m, n, k, gran_mn_a,
                             gran_k_a, gran_mn_b, gran_k_b, c.has_value());
        return;
    }

    if (try_cublaslt_fp8_c128_gemm(a, sfa, b, sfb, d, m, n, k, gran_mn_a,
                                   gran_mn_b, major_a, major_b,
                                   c.has_value()))
        return;

    const auto a_bf16 = dequant_fp8_c128_to_bf16(a, sfa, m, k, gran_mn_a, gran_k_a);
    const auto b_bf16 = dequant_fp8_c128_to_bf16(b, sfb, n, k, gran_mn_b, gran_k_b);

    cublaslt_gemm(a_bf16, b_bf16, d, m, n, k,
                  cute::UMMA::Major::K, cute::UMMA::Major::K,
                  c.has_value());
}

void sm120_fp8_bhr_hdr_bhd(const torch::Tensor& a,
                           const torch::Tensor& a_scale,
                           const torch::Tensor& b,
                           const torch::Tensor& b_scale,
                           const torch::Tensor& out) {
    DG_HOST_ASSERT(a.is_cuda() && a_scale.is_cuda() && b.is_cuda() &&
                   b_scale.is_cuda() && out.is_cuda());
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(out.scalar_type() == torch::kBFloat16 ||
                   out.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(a.dim() == 3 && out.dim() == 3);
    DG_HOST_ASSERT(b.dim() == 2 || b.dim() == 3);

    const int B = static_cast<int>(a.size(0));
    const int H = static_cast<int>(a.size(1));
    const int R = static_cast<int>(a.size(2));
    const int D = static_cast<int>(out.size(2));
    DG_HOST_ASSERT(out.size(0) == B && out.size(1) == H);
    if (b.dim() == 3) {
        DG_HOST_ASSERT(b.size(0) == H && b.size(1) == D && b.size(2) == R);
    } else {
        DG_HOST_ASSERT(b.size(0) == static_cast<int64_t>(H) * D &&
                       b.size(1) == R);
    }

    const int scale_blocks = (R + 127) / 128;
    auto scale_type = [](const torch::Tensor& scale) {
        if (scale.scalar_type() == torch::kFloat32)
            return 0;
        if (scale.scalar_type() == torch::kInt32)
            return 1;
        DG_HOST_UNREACHABLE("SM120 FP8 einsum scales must be float32 or int32");
    };
    const int a_scale_type = scale_type(a_scale);
    const int b_scale_type = scale_type(b_scale);
    const int packed_blocks = (scale_blocks + 3) / 4;
    const int a_expected_last = a_scale_type == 0 ? scale_blocks : packed_blocks;
    const int b_expected_last = b_scale_type == 0 ? scale_blocks : packed_blocks;

    DG_HOST_ASSERT(a_scale.size(-1) == a_expected_last);
    DG_HOST_ASSERT(b_scale.size(-1) == b_expected_last);
    DG_HOST_ASSERT(a_scale.dim() == 2 || a_scale.dim() == 3);
    DG_HOST_ASSERT(b_scale.dim() == 2 || b_scale.dim() == 3);

    const bool a_scale_flat = a_scale.dim() == 2;
    const bool b_scale_flat = b_scale.dim() == 2;
    if (a_scale_flat) {
        DG_HOST_ASSERT(a_scale.size(0) == static_cast<int64_t>(B) * H);
    } else {
        DG_HOST_ASSERT(a_scale.size(0) == B && a_scale.size(1) == H);
    }
    if (b_scale_flat) {
        DG_HOST_ASSERT(b_scale.size(0) == static_cast<int64_t>(H) * D ||
                       b_scale.size(0) ==
                           static_cast<int64_t>(H) * ((D + 127) / 128));
    } else {
        DG_HOST_ASSERT(b_scale.size(0) == H &&
                       (b_scale.size(1) == D ||
                        b_scale.size(1) == (D + 127) / 128));
    }

    const int b_scale_d_size =
        b_scale_flat ? static_cast<int>(b_scale.size(0) / H)
                     : static_cast<int>(b_scale.size(1));
    const int b_scale_gran_d = b_scale_d_size == D ? 1 : 128;

    const int64_t as_s0 = a_scale_flat ? H * a_scale.stride(0) : a_scale.stride(0);
    const int64_t as_s1 = a_scale_flat ? a_scale.stride(0) : a_scale.stride(1);
    const int64_t as_s2 = a_scale_flat ? a_scale.stride(1) : a_scale.stride(2);
    const int64_t bs_s0 = b_scale_flat ? b_scale.stride(0) : b_scale.stride(0);
    const int64_t bs_s1 = b_scale_flat ? b_scale.stride(1) : b_scale.stride(1);
    const int64_t bs_s2 = b_scale_flat ? b_scale.stride(1) : b_scale.stride(2);

            const bool use_m1_mma =
                env_flag_enabled("DG_SM120_ENABLE_BHR_M1_MMA") &&
                out.scalar_type() == torch::kBFloat16 && a_scale_type == 1 &&
                b_scale_type == 1 && R % 128 == 0 && D % 8 == 0 &&
                b_scale_gran_d == 128;
            const bool use_warp_col =
                env_flag_enabled("DG_SM120_ENABLE_BHR_WARPCOL") ||
                (env_flag_enabled("DG_SM120_ENABLE_BHR_WARPCOL_HEURISTIC") &&
                 R >= 2048 && D <= 4096 && B * H <= 64);
		    const bool use_kblock =
		        std::getenv("DG_SM120_BHR_KBLOCK_SCALE") != nullptr &&
		        R <= 512 && B * H <= 64 && !use_m1_mma && !use_warp_col;
		    const int threads =
		        use_kblock ? env_int_or_default("DG_SM120_BHR_KBLOCK_THREADS", 128)
		                   : 256;
		    int d_per_block = env_int_or_default("DG_SM120_BHR_D_PER_BLOCK", 1);
	    if (d_per_block != 1 && d_per_block != 2 && d_per_block != 4 &&
	        d_per_block != 8 && d_per_block != 16) {
	        d_per_block = 1;
	    }
	    const auto stream = at::cuda::getCurrentCUDAStream();
        static sm120_profile::KernelProfileCounter profile_counter(
            "sm120_fp8_bhr_hdr_bhd");
        sm120_profile::ScopedTimer profile_timer(
            profile_counter, stream, B, D, R, H);
        if (use_m1_mma) {
            constexpr int m1_warps_per_block = 4;
            const int grid =
                B * H * ((D + m1_warps_per_block * 8 - 1) /
                         (m1_warps_per_block * 8));
            fp8_bhr_hdr_bhd_m1_mma_kernel<<<grid, m1_warps_per_block * 32, 0,
                                             stream>>>(
                reinterpret_cast<const uint8_t*>(a.data_ptr()),
                static_cast<const int32_t*>(a_scale.data_ptr()),
                reinterpret_cast<const uint8_t*>(b.data_ptr()),
                static_cast<const int32_t*>(b_scale.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), B, H, D, R,
                b.dim(), b_scale.dim(), b_scale_d_size, b_scale_gran_d,
                a.stride(0), a.stride(1), a.stride(2), b.stride(0),
                b.stride(1), b.dim() == 3 ? b.stride(2) : 0, out.stride(0),
                out.stride(1), out.stride(2), as_s0, as_s1, as_s2, bs_s0,
                bs_s1, bs_s2);
            DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
            return;
        }
        if (use_warp_col) {
            int warps_per_block =
                env_int_or_default("DG_SM120_BHR_WARPCOL_WARPS", 4);
            if (warps_per_block != 8)
                warps_per_block = 4;
            int cols_per_warp =
                env_int_or_default("DG_SM120_BHR_WARPCOL_COLS", 1);
            if (cols_per_warp != 1 && cols_per_warp != 2 &&
                cols_per_warp != 4) {
                cols_per_warp = 2;
            }
            const int d_groups = (D + cols_per_warp - 1) / cols_per_warp;
            const int grid =
                (B * H * d_groups + warps_per_block - 1) / warps_per_block;
#define DG_SM120_LAUNCH_BHR_WARPCOL(OUT_T, OUT_PTR, WARPS, COLS)        \
            do {                                                        \
                fp8_bhr_hdr_bhd_warp_multi_col_kernel<OUT_T, WARPS, COLS> \
                    <<<grid, WARPS * 32, 0, stream>>>(                  \
                        reinterpret_cast<const uint8_t*>(a.data_ptr()),  \
                        a_scale.data_ptr(),                             \
                        reinterpret_cast<const uint8_t*>(b.data_ptr()),  \
                        b_scale.data_ptr(), OUT_PTR, B, H, D, R,        \
                        b.dim(), b_scale.dim(), b_scale_d_size,         \
                        b_scale_gran_d, a_scale_type, b_scale_type,     \
                        a.stride(0), a.stride(1), a.stride(2),          \
                        b.stride(0), b.stride(1),                       \
                        b.dim() == 3 ? b.stride(2) : 0, out.stride(0),  \
                        out.stride(1), out.stride(2), as_s0, as_s1,     \
                        as_s2, bs_s0, bs_s1, bs_s2);                    \
            } while (0)
#define DG_SM120_LAUNCH_BHR_WARPSINGLE(OUT_T, OUT_PTR, WARPS)           \
            do {                                                        \
                fp8_bhr_hdr_bhd_warp_single_col_kernel<OUT_T, WARPS>    \
                    <<<grid, WARPS * 32, 0, stream>>>(                  \
                        reinterpret_cast<const uint8_t*>(a.data_ptr()),  \
                        a_scale.data_ptr(),                             \
                        reinterpret_cast<const uint8_t*>(b.data_ptr()),  \
                        b_scale.data_ptr(), OUT_PTR, B, H, D, R,        \
                        b.dim(), b_scale.dim(), b_scale_d_size,         \
                        b_scale_gran_d, a_scale_type, b_scale_type,     \
                        a.stride(0), a.stride(1), a.stride(2),          \
                        b.stride(0), b.stride(1),                       \
                        b.dim() == 3 ? b.stride(2) : 0, out.stride(0),  \
                        out.stride(1), out.stride(2), as_s0, as_s1,     \
                        as_s2, bs_s0, bs_s1, bs_s2);                    \
            } while (0)
            if (out.scalar_type() == torch::kBFloat16) {
                auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
                if (warps_per_block == 8) {
                    if (cols_per_warp == 4) {
                        DG_SM120_LAUNCH_BHR_WARPCOL(__nv_bfloat16, out_ptr, 8, 4);
                    } else if (cols_per_warp == 2) {
                        DG_SM120_LAUNCH_BHR_WARPCOL(__nv_bfloat16, out_ptr, 8, 2);
                    } else {
                        DG_SM120_LAUNCH_BHR_WARPSINGLE(__nv_bfloat16, out_ptr, 8);
                    }
                } else {
                    if (cols_per_warp == 4) {
                        DG_SM120_LAUNCH_BHR_WARPCOL(__nv_bfloat16, out_ptr, 4, 4);
                    } else if (cols_per_warp == 2) {
                        DG_SM120_LAUNCH_BHR_WARPCOL(__nv_bfloat16, out_ptr, 4, 2);
                    } else {
                        DG_SM120_LAUNCH_BHR_WARPSINGLE(__nv_bfloat16, out_ptr, 4);
                    }
                }
            } else {
                auto* out_ptr = out.data_ptr<float>();
                if (warps_per_block == 8) {
                    if (cols_per_warp == 4) {
                        DG_SM120_LAUNCH_BHR_WARPCOL(float, out_ptr, 8, 4);
                    } else if (cols_per_warp == 2) {
                        DG_SM120_LAUNCH_BHR_WARPCOL(float, out_ptr, 8, 2);
                    } else {
                        DG_SM120_LAUNCH_BHR_WARPSINGLE(float, out_ptr, 8);
                    }
                } else {
                    if (cols_per_warp == 4) {
                        DG_SM120_LAUNCH_BHR_WARPCOL(float, out_ptr, 4, 4);
                    } else if (cols_per_warp == 2) {
                        DG_SM120_LAUNCH_BHR_WARPCOL(float, out_ptr, 4, 2);
                    } else {
                        DG_SM120_LAUNCH_BHR_WARPSINGLE(float, out_ptr, 4);
                    }
                }
            }
#undef DG_SM120_LAUNCH_BHR_WARPCOL
#undef DG_SM120_LAUNCH_BHR_WARPSINGLE
            DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
            return;
        }
	#define DG_SM120_LAUNCH_BHR(OUT_T, OUT_PTR, D_PER_BLOCK)                         \
	    do {                                                                         \
	        constexpr int kDPerBlockLaunch = D_PER_BLOCK;                            \
	        const int grid =                                                         \
	            B * H * ((D + kDPerBlockLaunch - 1) / kDPerBlockLaunch);             \
	        if (use_kblock) {                                                        \
	            fp8_bhr_hdr_bhd_kblock_scale_kernel<OUT_T, kDPerBlockLaunch>         \
	                <<<grid, threads, 0, stream>>>(                                  \
	                    reinterpret_cast<const uint8_t*>(a.data_ptr()),              \
	                    a_scale.data_ptr(),                                          \
	                    reinterpret_cast<const uint8_t*>(b.data_ptr()),              \
	                    b_scale.data_ptr(), OUT_PTR, B, H, D, R, b.dim(),            \
	                    b_scale.dim(), b_scale_d_size, b_scale_gran_d,               \
	                    a_scale_type, b_scale_type, a.stride(0), a.stride(1),        \
	                    a.stride(2), b.stride(0), b.stride(1),                       \
	                    b.dim() == 3 ? b.stride(2) : 0, out.stride(0),               \
	                    out.stride(1), out.stride(2), as_s0, as_s1, as_s2, bs_s0,    \
	                    bs_s1, bs_s2);                                               \
	        } else {                                                                 \
	            fp8_bhr_hdr_bhd_kernel<OUT_T, kDPerBlockLaunch>                      \
	                <<<grid, threads, 0, stream>>>(                                  \
	                    reinterpret_cast<const uint8_t*>(a.data_ptr()),              \
	                    a_scale.data_ptr(),                                          \
	                    reinterpret_cast<const uint8_t*>(b.data_ptr()),              \
	                    b_scale.data_ptr(), OUT_PTR, B, H, D, R, b.dim(),            \
	                    b_scale.dim(), b_scale_d_size, b_scale_gran_d,               \
	                    a_scale_type, b_scale_type, a.stride(0), a.stride(1),        \
	                    a.stride(2), b.stride(0), b.stride(1),                       \
	                    b.dim() == 3 ? b.stride(2) : 0, out.stride(0),               \
	                    out.stride(1), out.stride(2), as_s0, as_s1, as_s2, bs_s0,    \
	                    bs_s1, bs_s2);                                               \
	        }                                                                        \
	    } while (0)
	    if (out.scalar_type() == torch::kBFloat16) {
	        auto* out_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
	        switch (d_per_block) {
	            case 16: DG_SM120_LAUNCH_BHR(__nv_bfloat16, out_ptr, 16); break;
	            case 8: DG_SM120_LAUNCH_BHR(__nv_bfloat16, out_ptr, 8); break;
	            case 4: DG_SM120_LAUNCH_BHR(__nv_bfloat16, out_ptr, 4); break;
	            case 2: DG_SM120_LAUNCH_BHR(__nv_bfloat16, out_ptr, 2); break;
	            default: DG_SM120_LAUNCH_BHR(__nv_bfloat16, out_ptr, 1); break;
	        }
	    } else {
	        auto* out_ptr = out.data_ptr<float>();
	        switch (d_per_block) {
	            case 16: DG_SM120_LAUNCH_BHR(float, out_ptr, 16); break;
	            case 8: DG_SM120_LAUNCH_BHR(float, out_ptr, 8); break;
	            case 4: DG_SM120_LAUNCH_BHR(float, out_ptr, 4); break;
	            case 2: DG_SM120_LAUNCH_BHR(float, out_ptr, 2); break;
	            default: DG_SM120_LAUNCH_BHR(float, out_ptr, 1); break;
	        }
	    }
#undef DG_SM120_LAUNCH_BHR
	    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
	}

void sm120_m_grouped_fp8_fp4_gemm_nt_contiguous_fallback(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d, const torch::Tensor& grouped_layout,
    int num_groups, int m, int n, int k, int gran_k_a, int gran_k_b,
    const cute::UMMA::Major& major_b, bool use_psum_layout) {
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.scalar_type() == torch::kInt8);
    DG_HOST_ASSERT(sfa.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(sfb.scalar_type() == torch::kFloat32);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(grouped_layout.scalar_type() == torch::kInt32);
    DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);
    DG_HOST_ASSERT(gran_k_a == 128 && gran_k_b == 32);

    constexpr int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    const bool zero_padded_rows = std::getenv("DG_SM120_SKIP_PADDED_ZERO") == nullptr;

    auto launch_cols_4 = [&]() {
        constexpr int cols_per_block = 4;
        const int grid = m * ((n + cols_per_block - 1) / cols_per_block);
        if (use_psum_layout) {
            grouped_fp8_fp4_contiguous_kernel<__nv_bfloat16, true, cols_per_block>
                <<<grid, threads, 0, stream>>>(
                reinterpret_cast<const uint8_t*>(a.data_ptr()),
                sfa.data_ptr<float>(), b.data_ptr<int8_t>(),
                sfb.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(d.data_ptr()),
                grouped_layout.data_ptr<int32_t>(), num_groups, m, n, k,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                b.stride(2), d.stride(0), d.stride(1), sfa.stride(0),
                sfa.stride(1), sfb.stride(0), sfb.stride(1), sfb.stride(2),
                gran_k_a, gran_k_b, true, zero_padded_rows);
        } else {
            grouped_fp8_fp4_contiguous_kernel<__nv_bfloat16, false, cols_per_block>
                <<<grid, threads, 0, stream>>>(
                reinterpret_cast<const uint8_t*>(a.data_ptr()),
                sfa.data_ptr<float>(), b.data_ptr<int8_t>(),
                sfb.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(d.data_ptr()),
                grouped_layout.data_ptr<int32_t>(), num_groups, m, n, k,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                b.stride(2), d.stride(0), d.stride(1), sfa.stride(0),
                sfa.stride(1), sfb.stride(0), sfb.stride(1), sfb.stride(2),
                gran_k_a, gran_k_b, true, zero_padded_rows);
        }
    };

    auto launch_cols_8 = [&]() {
        constexpr int cols_per_block = 8;
        const int grid = m * ((n + cols_per_block - 1) / cols_per_block);
        if (use_psum_layout) {
            grouped_fp8_fp4_contiguous_kernel<__nv_bfloat16, true, cols_per_block>
                <<<grid, threads, 0, stream>>>(
                reinterpret_cast<const uint8_t*>(a.data_ptr()),
                sfa.data_ptr<float>(), b.data_ptr<int8_t>(),
                sfb.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(d.data_ptr()),
                grouped_layout.data_ptr<int32_t>(), num_groups, m, n, k,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                b.stride(2), d.stride(0), d.stride(1), sfa.stride(0),
                sfa.stride(1), sfb.stride(0), sfb.stride(1), sfb.stride(2),
                gran_k_a, gran_k_b, true, zero_padded_rows);
        } else {
            grouped_fp8_fp4_contiguous_kernel<__nv_bfloat16, false, cols_per_block>
                <<<grid, threads, 0, stream>>>(
                reinterpret_cast<const uint8_t*>(a.data_ptr()),
                sfa.data_ptr<float>(), b.data_ptr<int8_t>(),
                sfb.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(d.data_ptr()),
                grouped_layout.data_ptr<int32_t>(), num_groups, m, n, k,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                b.stride(2), d.stride(0), d.stride(1), sfa.stride(0),
                sfa.stride(1), sfb.stride(0), sfb.stride(1), sfb.stride(2),
                gran_k_a, gran_k_b, true, zero_padded_rows);
        }
    };

    auto launch_cols_16 = [&]() {
        constexpr int cols_per_block = 16;
        const int grid = m * ((n + cols_per_block - 1) / cols_per_block);
        if (use_psum_layout) {
            grouped_fp8_fp4_contiguous_kernel<__nv_bfloat16, true, cols_per_block>
                <<<grid, threads, 0, stream>>>(
                reinterpret_cast<const uint8_t*>(a.data_ptr()),
                sfa.data_ptr<float>(), b.data_ptr<int8_t>(),
                sfb.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(d.data_ptr()),
                grouped_layout.data_ptr<int32_t>(), num_groups, m, n, k,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                b.stride(2), d.stride(0), d.stride(1), sfa.stride(0),
                sfa.stride(1), sfb.stride(0), sfb.stride(1), sfb.stride(2),
                gran_k_a, gran_k_b, true, zero_padded_rows);
        } else {
            grouped_fp8_fp4_contiguous_kernel<__nv_bfloat16, false, cols_per_block>
                <<<grid, threads, 0, stream>>>(
                reinterpret_cast<const uint8_t*>(a.data_ptr()),
                sfa.data_ptr<float>(), b.data_ptr<int8_t>(),
                sfb.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(d.data_ptr()),
                grouped_layout.data_ptr<int32_t>(), num_groups, m, n, k,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                b.stride(2), d.stride(0), d.stride(1), sfa.stride(0),
                sfa.stride(1), sfb.stride(0), sfb.stride(1), sfb.stride(2),
                gran_k_a, gran_k_b, true, zero_padded_rows);
        }
    };

    const char* cols_env = std::getenv("DG_SM120_FP4_COLS_PER_BLOCK");
    if (cols_env != nullptr && std::strcmp(cols_env, "4") == 0) {
        launch_cols_4();
    } else if (cols_env != nullptr && std::strcmp(cols_env, "16") == 0) {
        launch_cols_16();
    } else {
        launch_cols_8();
    }
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

} // namespace deep_gemm
