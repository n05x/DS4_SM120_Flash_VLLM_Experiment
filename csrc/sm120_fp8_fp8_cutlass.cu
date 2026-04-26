#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/python.h>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cute/arch/mma_sm120.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

#include "jit_kernels/impls/sm120_fp8_fp8_cutlass.hpp"
#include "sm120_profile.hpp"
#include "utils/exception.hpp"

namespace deep_gemm {
namespace {

using namespace cute;

using ElementInputA = cutlass::float_e4m3_t;
using ElementInputB = cutlass::float_e4m3_t;
using ElementA = cutlass::mx_float8_t<ElementInputA>;
using ElementB = cutlass::mx_float8_t<ElementInputB>;
using ElementSF = cutlass::float_ue8m0_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementInputA>::value;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementInputB>::value;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using MxScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

struct DenseScratch {
    int device = -1;
    size_t sfa_capacity = 0;
    size_t sfb_capacity = 0;
    size_t workspace_capacity = 0;
    bool sfa_prefilled = false;
    int sfa_prefill_m = 0;
    int sfa_prefill_n = 0;
    int sfa_prefill_k = 0;
    size_t sfa_prefill_elems = 0;
    ElementSF* sfa = nullptr;
    ElementSF* sfb = nullptr;
    void* workspace = nullptr;
};

struct DenseSfbCacheEntry {
    int device = -1;
    uintptr_t data_ptr = 0;
    int scale_type = -1;
    int m = 0;
    int n = 0;
    int k = 0;
    int gran_mn = 0;
    int gran_k = 0;
    int64_t stride0 = 0;
    int64_t stride1 = 0;
    size_t elems = 0;
    ElementSF* data = nullptr;
};

std::mutex g_mutex;
DenseScratch g_scratch[16];
std::vector<DenseSfbCacheEntry> g_sfb_cache[16];

void cuda_check(cudaError_t status) {
    DG_HOST_ASSERT(status == cudaSuccess);
}

bool reserve_bytes(void*& ptr, size_t& capacity, size_t needed) {
    if (capacity >= needed)
        return false;
    if (ptr != nullptr)
        cuda_check(cudaFree(ptr));
    cuda_check(cudaMalloc(&ptr, needed));
    capacity = needed;
    return true;
}

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "False") != 0;
}

bool env_flag_default_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr || (std::strcmp(value, "0") != 0 &&
                                std::strcmp(value, "false") != 0 &&
                                std::strcmp(value, "False") != 0);
}

DenseScratch& get_scratch(int device, size_t sfa_bytes, size_t sfb_bytes,
                          size_t workspace_bytes) {
    std::lock_guard<std::mutex> lock(g_mutex);
    DenseScratch& s = g_scratch[device];
    s.device = device;
    if (reserve_bytes(reinterpret_cast<void*&>(s.sfa), s.sfa_capacity,
                      sfa_bytes)) {
        s.sfa_prefilled = false;
        s.sfa_prefill_elems = 0;
    }
    reserve_bytes(reinterpret_cast<void*&>(s.sfb), s.sfb_capacity, sfb_bytes);
    reserve_bytes(s.workspace, s.workspace_capacity, workspace_bytes);
    return s;
}

__device__ __forceinline__ ElementSF to_ue8m0(float x) {
    return ElementSF(x);
}

__global__ void fill_scale_kernel(ElementSF* __restrict__ out, size_t total) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total)
        out[idx] = ElementSF(1.0f);
}

void fill_scale_async(ElementSF* out, size_t total, cudaStream_t stream) {
    if (total == 0)
        return;
    if (env_flag_default_enabled("DG_SM120_USE_MEMSET_SCALE_FILL")) {
        // UE8M0 encodes 1.0 as exponent 127 (0x7f).  The padding entries in
        // CUTLASS' block-scale layouts only need the neutral multiplier, so a
        // byte memset avoids launching a separate fill kernel on every small-M
        // dense projection shape.
        cuda_check(cudaMemsetAsync(out, 0x7f, sizeof(ElementSF) * total,
                                   stream));
        return;
    }
    fill_scale_kernel<<<static_cast<unsigned>((total + 255) / 256), 256, 0,
                        stream>>>(out, total);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

__device__ __forceinline__ uint8_t load_ue8m0_raw(const int32_t* scale,
                                                  int64_t base,
                                                  int64_t stride_last,
                                                  int k_block) {
    const int32_t word =
        scale[base + static_cast<int64_t>(k_block / 4) * stride_last];
    return static_cast<uint8_t>((word >> ((k_block & 3) * 8)) & 0xff);
}

__global__ void convert_sfa_kernel(const void* __restrict__ sfa,
                                   ElementSF* __restrict__ out,
                                   int total, int scale_type, int m, int n,
                                   int k, int k32, int gran_mn, int gran_k,
                                   int64_t stride0, int64_t stride1) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int kk = idx % k32;
    const int row = idx / k32;
    const int k_block = kk / (gran_k / 32);
    const auto layout = MxScaleConfig::tile_atom_to_shape_SFA(
        make_shape(m, n, k, 1));
    const auto out_idx = layout(row, kk * 32, 0);
    const int64_t scale_base =
        static_cast<int64_t>(row / gran_mn) * stride0;
    if (scale_type == 0) {
        const auto* fp32 = static_cast<const float*>(sfa);
        out[out_idx] = to_ue8m0(
            fp32[scale_base + static_cast<int64_t>(k_block) * stride1]);
    } else {
        const auto* packed = static_cast<const int32_t*>(sfa);
        reinterpret_cast<uint8_t*>(out)[out_idx] =
            load_ue8m0_raw(packed, scale_base, stride1, k_block);
    }
}

__global__ void convert_sfb_kernel(const void* __restrict__ sfb,
                                   ElementSF* __restrict__ out,
                                   int total, int scale_type, int m, int n,
                                   int k, int k32, int gran_mn, int gran_k,
                                   int64_t stride0, int64_t stride1) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total)
        return;
    const int kk = idx % k32;
    const int col = idx / k32;
    const int k_block = kk / (gran_k / 32);
    const auto layout = MxScaleConfig::tile_atom_to_shape_SFB(
        make_shape(m, n, k, 1));
    const auto out_idx = layout(col, kk * 32, 0);
    const int64_t scale_base =
        static_cast<int64_t>(col / gran_mn) * stride0;
    if (scale_type == 0) {
        const auto* fp32 = static_cast<const float*>(sfb);
        out[out_idx] = to_ue8m0(
            fp32[scale_base + static_cast<int64_t>(k_block) * stride1]);
    } else {
        const auto* packed = static_cast<const int32_t*>(sfb);
        reinterpret_cast<uint8_t*>(out)[out_idx] =
            load_ue8m0_raw(packed, scale_base, stride1, k_block);
    }
}

ElementSF* get_cached_sfb(int device, const torch::Tensor& sfb, int scale_type,
                          int m, int n, int k, int k32, size_t sfb_elems,
                          int gran_mn, int gran_k, int64_t stride0,
                          int64_t stride1, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto& cache = g_sfb_cache[device];
    const uintptr_t data_ptr =
        reinterpret_cast<uintptr_t>(sfb.data_ptr());
    for (auto& entry : cache) {
        if (entry.device == device && entry.data_ptr == data_ptr &&
            entry.scale_type == scale_type && entry.m == m && entry.n == n &&
            entry.k == k && entry.gran_mn == gran_mn &&
            entry.gran_k == gran_k && entry.stride0 == stride0 &&
            entry.stride1 == stride1 && entry.elems == sfb_elems) {
            return entry.data;
        }
    }

    DenseSfbCacheEntry entry;
    entry.device = device;
    entry.data_ptr = data_ptr;
    entry.scale_type = scale_type;
    entry.m = m;
    entry.n = n;
    entry.k = k;
    entry.gran_mn = gran_mn;
    entry.gran_k = gran_k;
    entry.stride0 = stride0;
    entry.stride1 = stride1;
    entry.elems = sfb_elems;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&entry.data),
                          sizeof(ElementSF) * sfb_elems));

    fill_scale_async(entry.data, sfb_elems, stream);
    const int sfb_total = n * k32;
    convert_sfb_kernel<<<(sfb_total + 255) / 256, 256, 0, stream>>>(
        sfb.data_ptr(), entry.data, sfb_total, scale_type, m, n, k, k32,
        gran_mn, gran_k, stride0, stride1);
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    ElementSF* ptr = entry.data;
    cache.push_back(entry);
    return ptr;
}

int scale_tensor_type(const torch::Tensor& scale) {
    if (scale.scalar_type() == torch::kFloat32)
        return 0;
    if (scale.scalar_type() == torch::kInt32)
        return 1;
    return -1;
}

bool is_packed_k_major(const torch::Tensor& x, int rows, int cols) {
    return x.dim() == 2 && x.size(0) == rows && x.size(1) == cols &&
           x.stride(1) == 1 && x.stride(0) >= cols;
}

__device__ __forceinline__ void store_bf16(ElementD* __restrict__ d,
                                           int64_t offset, float value,
                                           bool accumulate) {
    auto* out = reinterpret_cast<__nv_bfloat16*>(d);
    if (accumulate)
        value += __bfloat162float(out[offset]);
    out[offset] = __float2bfloat16(value);
}

__global__ void fp8_fp8_m1_mma_kernel(
    const uint8_t* __restrict__ a, const int32_t* __restrict__ sfa,
    const uint8_t* __restrict__ b, const int32_t* __restrict__ sfb,
    ElementD* __restrict__ d, int n, int k, int k32,
    int gran_mn_b, int gran_k_a, int gran_k_b,
    int64_t a_s1, int64_t b_s0, int64_t b_s1, int64_t d_s1,
    int64_t sfa_s1, int64_t sfb_s0, int64_t sfb_s1,
    bool accumulate) {
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
    const int col_base = (blockIdx.x * kWarpsPerBlock + warp_id) * 8;
    if (col_base >= n)
        return;

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
            if (local_row == 0)
                value = a[static_cast<int64_t>(kb * 32 + kk) * a_s1];
            a_reg[v >> 2] |= static_cast<uint32_t>(value) << ((v & 3) * 8);
        }

#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int v0 = v & 3;
            const int v1 = v >> 2;
            const int offset = t0 * 32 + t1 + v0 * 8 + v1 * 128;
            const int kk = offset >> 3;
            const int n_local = offset & 7;
            const int col = col_base + n_local;
            uint8_t value = 0;
            if (col < n) {
                value = b[static_cast<int64_t>(col) * b_s0 +
                          static_cast<int64_t>(kb * 32 + kk) * b_s1];
            }
            b_reg[v >> 2] |= static_cast<uint32_t>(value) << ((v & 3) * 8);
        }

        const int a_scale_block = kb / (gran_k_a / 32);
        const uint8_t sfa_value = load_ue8m0_raw(sfa, 0, sfa_s1, a_scale_block);
        const int scale_col = min(col_base + t1, n - 1) / gran_mn_b;
        const int b_scale_block = kb / (gran_k_b / 32);
        const uint8_t sfb_value =
            load_ue8m0_raw(sfb, static_cast<int64_t>(scale_col) * sfb_s0,
                           sfb_s1, b_scale_block);

        MmaOp::fma(acc[0], acc[1], acc[2], acc[3],
                   a_reg[0], a_reg[1], a_reg[2], a_reg[3],
                   b_reg[0], b_reg[1],
                   acc[0], acc[1], acc[2], acc[3],
                   sfa_value, sfb_value);
    }

    if (t1 == 0) {
        const int col0 = col_base + t0 * 2;
        if (col0 < n)
            store_bf16(d, static_cast<int64_t>(col0) * d_s1, acc[0],
                       accumulate);
        const int col1 = col0 + 1;
        if (col1 < n)
            store_bf16(d, static_cast<int64_t>(col1) * d_s1, acc[1],
                       accumulate);
    }
#endif
}

} // namespace

bool sm120_fp8_fp8_gemm_nt_cutlass(
    const torch::Tensor& a, const torch::Tensor& sfa,
    const torch::Tensor& b, const torch::Tensor& sfb,
    const torch::Tensor& d, int m, int n, int k,
    int gran_mn_a, int gran_k_a, int gran_mn_b, int gran_k_b,
    const cute::UMMA::Major& major_a, const cute::UMMA::Major& major_b,
    bool accumulate) {
    if (std::getenv("DG_SM120_DISABLE_CUTLASS_FP8") != nullptr)
        return false;
    if (major_a != cute::UMMA::Major::K ||
        major_b != cute::UMMA::Major::K ||
        gran_k_a != 128 || gran_k_b != 128 || k % 128 != 0 ||
        n % 128 != 0 || d.scalar_type() != torch::kBFloat16) {
        return false;
    }
    if (scale_tensor_type(sfa) < 0 || scale_tensor_type(sfb) < 0)
        return false;
    if (!is_packed_k_major(a, m, k) || !is_packed_k_major(b, n, k))
        return false;
    if (m == 1 && sfa.scalar_type() == torch::kInt32 &&
        sfb.scalar_type() == torch::kInt32 &&
        env_flag_enabled("DG_SM120_ENABLE_FP8_M1_MMA")) {
        const auto stream = at::cuda::getCurrentCUDAStream();
        static sm120_profile::KernelProfileCounter profile_counter(
            "sm120_fp8_fp8_m1_mma");
        sm120_profile::ScopedTimer profile_timer(
            profile_counter, stream, m, n, k, 1);
        constexpr int m1_warps_per_block = 4;
        fp8_fp8_m1_mma_kernel<<<
            (n + m1_warps_per_block * 8 - 1) / (m1_warps_per_block * 8),
            m1_warps_per_block * 32, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(a.data_ptr()),
            sfa.data_ptr<int32_t>(),
            reinterpret_cast<const uint8_t*>(b.data_ptr()),
            sfb.data_ptr<int32_t>(), reinterpret_cast<ElementD*>(d.data_ptr()),
            n, k, k / 32, gran_mn_b, gran_k_a, gran_k_b, a.stride(1),
            b.stride(0), b.stride(1), d.stride(1), sfa.stride(1),
            sfb.stride(0), sfb.stride(1), accumulate);
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
        return true;
    }
    if (m < 2)
        return false;

    const int k32 = k / 32;
    const auto layout_sfa =
        MxScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
    const auto layout_sfb =
        MxScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));
    const size_t sfa_elems = static_cast<size_t>(size(filter_zeros(layout_sfa)));
    const size_t sfb_elems = static_cast<size_t>(size(filter_zeros(layout_sfb)));
    const int sfa_type = scale_tensor_type(sfa);
    const int sfb_type = scale_tensor_type(sfb);
    const bool cache_sfb =
        env_flag_default_enabled("DG_SM120_CACHE_FP8_SFB");

    StrideA stride_a{a.stride(0), cute::Int<1>{}, 0};
    StrideB stride_b{b.stride(0), cute::Int<1>{}, 0};
    StrideC stride_c{d.stride(0), cute::Int<1>{}, 0};
    StrideD stride_d{d.stride(0), cute::Int<1>{}, 0};

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        {reinterpret_cast<ElementInputA const*>(a.data_ptr()), stride_a,
         reinterpret_cast<ElementInputB const*>(b.data_ptr()), stride_b,
         nullptr, layout_sfa, nullptr, layout_sfb},
        {{1.0f, accumulate ? 1.0f : 0.0f},
         reinterpret_cast<ElementC const*>(d.data_ptr()), stride_c,
         reinterpret_cast<ElementD*>(d.data_ptr()), stride_d}};

    const size_t workspace_size = Gemm::get_workspace_size(arguments);
    const int device = a.get_device();
    DenseScratch& scratch =
        get_scratch(device, sizeof(ElementSF) * sfa_elems,
                    cache_sfb ? 0 : sizeof(ElementSF) * sfb_elems,
                    workspace_size);
    arguments.mainloop.ptr_SFA = scratch.sfa;

    const auto stream = at::cuda::getCurrentCUDAStream();
    static sm120_profile::KernelProfileCounter profile_counter(
        "sm120_fp8_fp8_cutlass");
    sm120_profile::ScopedTimer profile_timer(
        profile_counter, stream, m, n, k, 1);
    if (cache_sfb) {
        arguments.mainloop.ptr_SFB =
            get_cached_sfb(device, sfb, sfb_type, m, n, k, k32, sfb_elems,
                           gran_mn_b, gran_k_b, sfb.stride(0),
                           sfb.stride(1), stream);
    } else {
        arguments.mainloop.ptr_SFB = scratch.sfb;
        fill_scale_async(scratch.sfb, sfb_elems, stream);
        const int sfb_total = n * k32;
        convert_sfb_kernel<<<(sfb_total + 255) / 256, 256, 0, stream>>>(
            sfb.data_ptr(), scratch.sfb, sfb_total, sfb_type, m, n, k, k32,
            gran_mn_b, gran_k_b, sfb.stride(0), sfb.stride(1));
        DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
    }

    const bool cache_sfa_fill =
        env_flag_default_enabled("DG_SM120_CACHE_FP8_SFA_FILL");
    const bool need_sfa_prefill =
        !cache_sfa_fill || !scratch.sfa_prefilled ||
        scratch.sfa_prefill_m != m || scratch.sfa_prefill_n != n ||
        scratch.sfa_prefill_k != k ||
        scratch.sfa_prefill_elems != sfa_elems;
    if (need_sfa_prefill) {
        fill_scale_async(scratch.sfa, sfa_elems, stream);
        scratch.sfa_prefilled = cache_sfa_fill;
        scratch.sfa_prefill_m = m;
        scratch.sfa_prefill_n = n;
        scratch.sfa_prefill_k = k;
        scratch.sfa_prefill_elems = sfa_elems;
    }

    const int sfa_total = m * k32;
    convert_sfa_kernel<<<(sfa_total + 255) / 256, 256, 0, stream>>>(
        sfa.data_ptr(), scratch.sfa, sfa_total, sfa_type, m, n, k, k32,
        gran_mn_a, gran_k_a, sfa.stride(0), sfa.stride(1));
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess)
        return false;
    status = gemm_op.initialize(arguments, scratch.workspace, stream);
    if (status != cutlass::Status::kSuccess)
        return false;
    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess)
        return false;
    DG_CUDA_RUNTIME_CHECK(cudaGetLastError());
    return true;
}

} // namespace deep_gemm
