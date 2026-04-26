#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <cuda_runtime.h>

namespace deep_gemm::sm120_profile {

struct KernelProfileCounter {
    explicit KernelProfileCounter(const char* kernel_name) : name(kernel_name) {}

    const char* name;
    std::mutex mutex;
    unsigned long long calls = 0;
    double total_ms = 0.0;
    double max_ms = 0.0;
};

inline bool enabled() {
    const char* value = std::getenv("DG_SM120_KERNEL_PROFILE");
    return value != nullptr && std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "False") != 0;
}

inline int interval() {
    const char* value = std::getenv("DG_SM120_KERNEL_PROFILE_EVERY");
    if (value == nullptr)
        return 256;
    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : 256;
}

inline const char* output_path() {
    const char* value = std::getenv("DG_SM120_KERNEL_PROFILE_PATH");
    return value != nullptr && value[0] != '\0'
               ? value
               : "/tmp/dg_sm120_kernel_profile.tsv";
}

inline void record(KernelProfileCounter& counter, float ms, int m, int n, int k,
                   int groups) {
    std::lock_guard<std::mutex> lock(counter.mutex);
    counter.calls += 1;
    counter.total_ms += static_cast<double>(ms);
    if (ms > counter.max_ms)
        counter.max_ms = ms;

    const int every = interval();
    if (counter.calls % static_cast<unsigned long long>(every) != 0)
        return;

    FILE* file = std::fopen(output_path(), "a");
    if (file == nullptr)
        return;
    std::fprintf(file,
                 "%s\tcalls=%llu\tavg_ms=%.6f\tmax_ms=%.6f\tlast_ms=%.6f"
                 "\tm=%d\tn=%d\tk=%d\tgroups=%d\n",
                 counter.name, counter.calls,
                 counter.total_ms / static_cast<double>(counter.calls),
                 counter.max_ms, static_cast<double>(ms), m, n, k, groups);
    std::fclose(file);
}

class ScopedTimer {
public:
    ScopedTimer(KernelProfileCounter& counter, cudaStream_t stream, int m, int n,
                int k, int groups)
        : counter_(counter), stream_(stream), m_(m), n_(n), k_(k),
          groups_(groups), active_(enabled()) {
        if (!active_)
            return;
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        const cudaError_t capture_error =
            cudaStreamIsCapturing(stream_, &capture_status);
        if (capture_error == cudaSuccess &&
            capture_status != cudaStreamCaptureStatusNone) {
            active_ = false;
            return;
        }
        if (capture_error != cudaSuccess) {
            cudaGetLastError();
            active_ = false;
            return;
        }
        if (cudaEventCreate(&start_) != cudaSuccess ||
            cudaEventCreate(&stop_) != cudaSuccess) {
            active_ = false;
            return;
        }
        cudaEventRecord(start_, stream_);
    }

    ~ScopedTimer() {
        if (!active_)
            return;
        cudaEventRecord(stop_, stream_);
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, start_, stop_) == cudaSuccess)
            record(counter_, ms, m_, n_, k_, groups_);
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    KernelProfileCounter& counter_;
    cudaStream_t stream_;
    int m_;
    int n_;
    int k_;
    int groups_;
    bool active_ = false;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

} // namespace deep_gemm::sm120_profile
