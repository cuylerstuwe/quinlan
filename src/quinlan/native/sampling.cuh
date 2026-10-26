// Quinlan Qwentin sampling overlay: GPU sampling and the MTP probability verifier.
// GPU logits and all filtering stay on device. Called OUTSIDE captured model graphs.
// Filter order: temperature -> top-k -> top-p (keep crossing token) -> min-p.
// Greedy MTP is preserved at T=0. Stochastic MTP has exact probability-based acceptance.
#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/cub.cuh>
#include <stdint.h>
#include <math.h>
#include <limits>

struct quinlan_sampling_params {
    float temperature = 0, top_p = 1, min_p = 0;
    int top_k = 0, origin = 0;
    uint64_t seed = 0;
};
static quinlan_sampling_params quinlan_params[256];
static float *quinlan_keys_in = nullptr, *quinlan_keys_out = nullptr;
static float *quinlan_weights = nullptr, *quinlan_cdf = nullptr;
static int *quinlan_ids_in = nullptr, *quinlan_ids_out = nullptr, *quinlan_bad = nullptr;
static void *quinlan_temp = nullptr;
static size_t quinlan_temp_bytes = 0;
static int quinlan_capacity = 0;
static int *quinlan_cut = nullptr, *quinlan_accept = nullptr;
static float *quinlan_norm = nullptr, *quinlan_p = nullptr, *quinlan_q = nullptr;
static size_t quinlan_q_capacity = 0;
static int quinlan_q_N = 0, quinlan_q_K = 0, quinlan_q_V = 0;
static int quinlan_q_slots[256], quinlan_q_positions[256];
static int quinlan_spec_mode = 0, quinlan_spec_round = 0;
extern "C" void quinlan_spec_set_mode(int mode) { quinlan_spec_mode = mode != 0; }

extern "C" int quinlan_sampling_configure(int slot, float temperature, float top_p,
                                      int top_k, float min_p, uint64_t seed, int origin) {
    if (slot < 0 || slot >= 256 || !isfinite(temperature) || temperature < 0 ||
        !isfinite(top_p) || top_p <= 0 || top_p > 1 || top_k < 0 ||
        !isfinite(min_p) || min_p < 0 || min_p > 1 || origin < 0) return -120;
    quinlan_params[slot] = {temperature, top_p, min_p, top_k, origin, seed};
    return 0;
}

static void quinlan_sampling_free() {
    cudaFree(quinlan_keys_in); cudaFree(quinlan_keys_out); cudaFree(quinlan_weights); cudaFree(quinlan_cdf);
    cudaFree(quinlan_ids_in); cudaFree(quinlan_ids_out); cudaFree(quinlan_bad); cudaFree(quinlan_temp);
    cudaFree(quinlan_cut); cudaFree(quinlan_accept); cudaFree(quinlan_norm); cudaFree(quinlan_p);
    cudaFree(quinlan_q); quinlan_q = nullptr; quinlan_q_capacity = 0;
    quinlan_q_N = quinlan_q_K = quinlan_q_V = 0;
    quinlan_keys_in = quinlan_keys_out = quinlan_weights = quinlan_cdf = nullptr;
    quinlan_ids_in = quinlan_ids_out = quinlan_bad = nullptr; quinlan_temp = nullptr;
    quinlan_capacity = 0; quinlan_temp_bytes = 0;
    quinlan_cut = quinlan_accept = nullptr; quinlan_norm = quinlan_p = nullptr;
}

static int quinlan_sampling_alloc(int V, cudaStream_t stream) {
    if (quinlan_capacity >= V) return 0;
    if (cudaStreamSynchronize(stream) != cudaSuccess) return -121;
    quinlan_sampling_free();
    if (cudaMalloc(&quinlan_keys_in, V * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&quinlan_keys_out, V * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&quinlan_weights, V * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&quinlan_cdf, V * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&quinlan_ids_in, V * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&quinlan_ids_out, V * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&quinlan_bad, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&quinlan_cut, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&quinlan_accept, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&quinlan_norm, sizeof(float)) != cudaSuccess ||
        cudaMalloc(&quinlan_p, V * sizeof(float)) != cudaSuccess) {
        quinlan_sampling_free(); return -122;
    }
    size_t sort_bytes = 0, scan_bytes = 0;
    if (cub::DeviceRadixSort::SortPairsDescending(nullptr, sort_bytes, quinlan_keys_in, quinlan_keys_out,
            quinlan_ids_in, quinlan_ids_out, V, 0, 32, stream) != cudaSuccess ||
        cub::DeviceScan::InclusiveSum(nullptr, scan_bytes, quinlan_weights, quinlan_cdf, V, stream) != cudaSuccess)
        { quinlan_sampling_free(); return -123; }
    quinlan_temp_bytes = sort_bytes > scan_bytes ? sort_bytes : scan_bytes;
    if (cudaMalloc(&quinlan_temp, quinlan_temp_bytes) != cudaSuccess) { quinlan_sampling_free(); return -122; }
    quinlan_capacity = V;
    return 0;
}

__global__ void quinlan_sampling_prepare(const float *logits, float *keys, int *ids, int *bad, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V) return;
    float x = logits[i];
    // -infinity is a valid mask; NaN and +infinity indicate a numerical failure.
    if (isnan(x) || x == CUDART_INF_F) atomicExch(bad, 1);
    keys[i] = isnan(x) ? -CUDART_INF_F : x;
    ids[i] = i;
}

__global__ void quinlan_sampling_weights(const float *sorted, float *weights, int V, int top_k, float T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= V) return;
    weights[i] = (top_k > 0 && i >= top_k) || !isfinite(sorted[0]) || sorted[i] == -CUDART_INF_F
        ? 0.0f : expf((sorted[i] - sorted[0]) / T);
}

__host__ __device__ static uint64_t quinlan_mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__global__ void quinlan_sampling_pick(int *out, const int *ids, const float *weights, const float *cdf,
                                int *bad, int V, float top_p, float min_p, uint64_t seed, int index,
                                const float *map, int *end_out, float *sum_out) {
    if (threadIdx.x || blockIdx.x) return;
    float total = cdf[V - 1];
    if (!(total > 0) || !isfinite(total) || index < 0) { atomicExch(bad, 1); return; }
    // Nucleus contains the first token crossing the requested cumulative mass.
    int lo = 0, hi = V - 1;
    float target = top_p * total;
    while (lo < hi) { int m = lo + (hi - lo) / 2; if (cdf[m] >= target) hi = m; else lo = m + 1; }
    int end = lo;
    // min-p is relative to the most probable token, after top-p; keep equal ties.
    if (min_p > 0) {
        lo = 0; hi = end + 1;
        while (lo < hi) { int m = lo + (hi - lo) / 2; if (weights[m] >= min_p) lo = m + 1; else hi = m; }
        end = min(end, max(0, lo - 1));
    }
    uint64_t bits = quinlan_mix64(seed ^ quinlan_mix64((uint64_t)index));
    double u = ((double)(bits >> 11) + 0.5) * 0x1.0p-53;
    double threshold = u * (double)cdf[end];
    lo = 0; hi = end;
    while (lo < hi) { int m = lo + (hi - lo) / 2; if ((double)cdf[m] > threshold) hi = m; else lo = m + 1; }
    *out = map ? (int)map[ids[lo]] : ids[lo];
    if (end_out) *end_out = end;
    if (sum_out) *sum_out = cdf[end];
}

// Test ABI also drives the integrated paths. Caller serializes use on the engine thread.
extern "C" int quinlan_sampling_apply(int *out, const float *logits, int V, int N,
                                  const int *slots, const int *positions, cudaStream_t stream) {
    if (V < 1 || N < 1) return -124;
    bool enabled = false;
    for (int c = 0; c < N; c++) {
        if (slots[c] < 0 || slots[c] >= 256) return -120;
        enabled |= quinlan_params[slots[c]].temperature > 0;
    }
    if (!enabled) return 0; // preserve the original greedy argmax, no GPU launch or allocation
    int rc = quinlan_sampling_alloc(V, stream);
    if (rc) return rc;
    if (cudaMemsetAsync(quinlan_bad, 0, sizeof(int), stream) != cudaSuccess) return -121;
    for (int c = 0; c < N; c++) {
        const auto p = quinlan_params[slots[c]];
        if (p.temperature == 0) continue;
        size_t sort_bytes = quinlan_temp_bytes, scan_bytes = quinlan_temp_bytes;
        quinlan_sampling_prepare<<<(V + 255) / 256, 256, 0, stream>>>(logits + (size_t)c * V,
            quinlan_keys_in, quinlan_ids_in, quinlan_bad, V);
        if (cub::DeviceRadixSort::SortPairsDescending(quinlan_temp, sort_bytes, quinlan_keys_in, quinlan_keys_out,
                quinlan_ids_in, quinlan_ids_out, V, 0, 32, stream) != cudaSuccess) return -125;
        quinlan_sampling_weights<<<(V + 255) / 256, 256, 0, stream>>>(quinlan_keys_out, quinlan_weights, V, p.top_k, p.temperature);
        if (cub::DeviceScan::InclusiveSum(quinlan_temp, scan_bytes, quinlan_weights, quinlan_cdf, V, stream) != cudaSuccess) return -126;
        quinlan_sampling_pick<<<1, 1, 0, stream>>>(out + c, quinlan_ids_out, quinlan_weights, quinlan_cdf,
            quinlan_bad, V, p.top_p, p.min_p, p.seed, positions[c] + 1 - p.origin, nullptr, nullptr, nullptr);
    }
    int bad = 0;
    if (cudaGetLastError() != cudaSuccess || cudaMemcpyAsync(&bad, quinlan_bad, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) return -121;
    return bad ? -127 : 0;
}

// Distribution-preserving stochastic speculation. q is the ACTUAL filtered,
// normalized draft distribution, including the reduced-head token-ID mapping.
// Accept d with min(1,p[d]/q[d]); rejection draws from normalized max(p-q,0).
// Separate random domains for proposals, accept/reject and residual draws.
static constexpr uint64_t Quinlan_DRAFT_DOMAIN = 0x44524146545f524eULL;
static constexpr uint64_t Quinlan_ACCEPT_DOMAIN = 0x414343455054524eULL;
static constexpr uint64_t Quinlan_RESIDUAL_DOMAIN = 0x524553494455524eULL;

__global__ void quinlan_prob_scatter(float *dense, const int *ids, const float *weights,
                               const int *end, const float *norm, const float *map, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V && i <= *end && weights[i] > 0) {
        int id = map ? (int)map[ids[i]] : ids[i];
        dense[id] = weights[i] / *norm;
    }
}

static int quinlan_distribution(int *out, float *dense, const float *logits, int V, int full_V,
                           const float *map, const quinlan_sampling_params &p, int index,
                           uint64_t domain, cudaStream_t stream) {
    if (cudaMemsetAsync(dense, 0, full_V * sizeof(float), stream) != cudaSuccess) return -121;
    quinlan_sampling_prepare<<<(V + 255) / 256, 256, 0, stream>>>(logits, quinlan_keys_in, quinlan_ids_in, quinlan_bad, V);
    size_t sort_bytes = quinlan_temp_bytes, scan_bytes = quinlan_temp_bytes;
    if (cub::DeviceRadixSort::SortPairsDescending(quinlan_temp, sort_bytes, quinlan_keys_in, quinlan_keys_out,
            quinlan_ids_in, quinlan_ids_out, V, 0, 32, stream) != cudaSuccess) return -125;
    quinlan_sampling_weights<<<(V + 255) / 256, 256, 0, stream>>>(quinlan_keys_out, quinlan_weights, V, p.top_k, p.temperature);
    if (cub::DeviceScan::InclusiveSum(quinlan_temp, scan_bytes, quinlan_weights, quinlan_cdf, V, stream) != cudaSuccess) return -126;
    quinlan_sampling_pick<<<1, 1, 0, stream>>>(out, quinlan_ids_out, quinlan_weights, quinlan_cdf, quinlan_bad,
        V, p.top_p, p.min_p, p.seed ^ domain, index, map, quinlan_cut, quinlan_norm);
    quinlan_prob_scatter<<<(V + 255) / 256, 256, 0, stream>>>(dense, quinlan_ids_out, quinlan_weights,
        quinlan_cut, quinlan_norm, map, V);
    return 0;
}

static int quinlan_check(cudaStream_t stream) {
    int bad = 0;
    if (cudaGetLastError() != cudaSuccess || cudaMemcpyAsync(&bad, quinlan_bad, sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) return -121;
    return bad ? -127 : 0;
}

static int quinlan_draft_begin(const int *slots, const int *positions, int N, int K, int V, cudaStream_t stream) {
    quinlan_spec_round = 0;
    if (!quinlan_spec_mode) return 0;
    for (int c = 0; c < N; c++) quinlan_spec_round |= quinlan_params[slots[c]].temperature > 0;
    if (!quinlan_spec_round) return 0;
    int rc = quinlan_sampling_alloc(V, stream); if (rc) return rc;
    size_t capacity = (size_t)N * K * V;
    if (capacity > quinlan_q_capacity) {
        cudaFree(quinlan_q); quinlan_q = nullptr; quinlan_q_capacity = 0;
        if (cudaMalloc(&quinlan_q, capacity * sizeof(float)) != cudaSuccess) return -122;
        quinlan_q_capacity = capacity;
    }
    quinlan_q_N = N; quinlan_q_K = K; quinlan_q_V = V;
    for (int c = 0; c < N; c++) { quinlan_q_slots[c] = slots[c]; quinlan_q_positions[c] = positions[c]; }
    return cudaMemsetAsync(quinlan_bad, 0, sizeof(int), stream) == cudaSuccess ? 0 : -121;
}

static int quinlan_draft_step(int *out, const float *logits, const float *map, int V, int N, int k, cudaStream_t stream) {
    if (!quinlan_spec_round) return 0;
    for (int c = 0; c < N; c++) {
        const auto p = quinlan_params[quinlan_q_slots[c]];
        if (p.temperature == 0) continue;
        int index = quinlan_q_positions[c] + k + 2 - p.origin;
        int rc = quinlan_distribution(out + c, quinlan_q + ((size_t)c * quinlan_q_K + k) * quinlan_q_V,
            logits + (size_t)c * V, V, quinlan_q_V, map, p, index, Quinlan_DRAFT_DOMAIN, stream);
        if (rc) return rc;
    }
    return quinlan_check(stream);
}

__global__ void quinlan_accept_decide(int *accept, int *bad, const float *p, const float *q,
                                int draft, int V, uint64_t seed, int index) {
    if (threadIdx.x || blockIdx.x) return;
    if (draft < 0 || draft >= V || !(q[draft] > 0) || !isfinite(q[draft]) || !isfinite(p[draft])) {
        atomicExch(bad, 1); *accept = 0; return;
    }
    double u = ((double)(quinlan_mix64(seed ^ Quinlan_ACCEPT_DOMAIN ^ quinlan_mix64((uint64_t)index)) >> 11) + .5) * 0x1.0p-53;
    *accept = u * (double)q[draft] < (double)p[draft];
}
__global__ void quinlan_residual_weights(float *weights, const float *p, const float *q, const int *accept, int V) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) weights[i] = *accept ? 0 : fmaxf(p[i] - q[i], 0.0f);
}
__global__ void quinlan_residual_pick(int *out, int *bad, const float *cdf, const int *accept,
                                int draft, int V, uint64_t seed, int index) {
    if (threadIdx.x || blockIdx.x) return;
    if (*accept) { *out = draft; return; }
    float total = cdf[V - 1];
    if (!(total > 0) || !isfinite(total)) { atomicExch(bad, 1); return; }
    double u = ((double)(quinlan_mix64(seed ^ Quinlan_RESIDUAL_DOMAIN ^ quinlan_mix64((uint64_t)index)) >> 11) + .5) * 0x1.0p-53;
    double threshold = u * (double)total;
    int lo = 0, hi = V - 1;
    while (lo < hi) { int m = lo + (hi - lo) / 2; if ((double)cdf[m] > threshold) hi = m; else lo = m + 1; }
    *out = lo;
}
static int quinlan_accept_row(int *out, const float *p, const float *q, int draft, int V,
                          uint64_t seed, int index, cudaStream_t stream) {
    quinlan_accept_decide<<<1, 1, 0, stream>>>(quinlan_accept, quinlan_bad, p, q, draft, V, seed, index);
    quinlan_residual_weights<<<(V + 255) / 256, 256, 0, stream>>>(quinlan_weights, p, q, quinlan_accept, V);
    size_t scan_bytes = quinlan_temp_bytes;
    if (cub::DeviceScan::InclusiveSum(quinlan_temp, scan_bytes, quinlan_weights, quinlan_cdf, V, stream) != cudaSuccess) return -126;
    quinlan_residual_pick<<<1, 1, 0, stream>>>(out, quinlan_bad, quinlan_cdf, quinlan_accept, draft, V, seed, index);
    return 0;
}

static int quinlan_spec_verify(int *out, const float *logits, int V, int N, const int *tokens,
                            const int *slots, const int *positions, cudaStream_t stream) {
    if (!quinlan_spec_round) return quinlan_sampling_apply(out, logits, V, N, slots, positions, stream);
    if (V != quinlan_q_V) return -129;
    if (cudaMemsetAsync(quinlan_bad, 0, sizeof(int), stream) != cudaSuccess) return -121;
    for (int c = 0; c < N; c++) {
        const auto p = quinlan_params[slots[c]];
        if (p.temperature == 0) continue;
        int qi = -1;
        for (int i = 0; i < quinlan_q_N; i++) if (quinlan_q_slots[i] == slots[c]) qi = i;
        if (qi < 0) return -129;
        int k = positions[c] - quinlan_q_positions[qi] - 1;
        if (k < 0 || k > quinlan_q_K) return -129;
        int index = positions[c] + 1 - p.origin;
        int rc = quinlan_distribution(out + c, quinlan_p, logits + (size_t)c * V, V, V, nullptr, p, index, 0, stream);
        if (rc) return rc;
        if (k < quinlan_q_K) {
            if (c + 1 >= N || slots[c+1] != slots[c] || positions[c+1] != positions[c] + 1) return -129;
            rc = quinlan_accept_row(out + c, quinlan_p, quinlan_q + ((size_t)qi * quinlan_q_K + k) * V,
                tokens[c+1], V, p.seed, index, stream);
            if (rc) return rc;
        }
    }
    return quinlan_check(stream);
}

// Narrow test ABI exercises exactly the acceptance/residual kernels used above.
extern "C" int quinlan_spec_test(int *out, const float *p, const float *q, const int *drafts,
                              int V, int N, uint64_t seed, cudaStream_t stream) {
    int rc = quinlan_sampling_alloc(V, stream); if (rc) return rc;
    if (cudaMemsetAsync(quinlan_bad, 0, sizeof(int), stream) != cudaSuccess) return -121;
    for (int c = 0; c < N; c++) {
        rc = quinlan_accept_row(out+c, p+(size_t)c*V, q+(size_t)c*V, drafts[c], V, seed, c, stream);
        if (rc) return rc;
    }
    return quinlan_check(stream);
}

extern "C" int quinlan_spec_pipeline_test(int *drafts, int *verified, const float *draft_logits,
    const float *target_logits, const float *map, int draft_V, int full_V, int K, cudaStream_t stream) {
    if (K < 1 || K > 16 || draft_V < 1 || draft_V > full_V) return -130;
    quinlan_spec_mode = 1;
    int slot = 0, committed = 0;
    int rc = quinlan_draft_begin(&slot, &committed, 1, K, full_V, stream); if (rc) return rc;
    int *d_out = nullptr;
    if (cudaMalloc(&d_out, (K + 1) * sizeof(int)) != cudaSuccess) return -122;
    for (int k = 0; k < K; k++) {
        rc = quinlan_draft_step(d_out+k, draft_logits+(size_t)k*draft_V, map, draft_V, 1, k, stream);
        if (rc) { cudaFree(d_out); return rc; }
    }
    if (cudaMemcpy(drafts, d_out, K*sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) { cudaFree(d_out); return -121; }
    int tokens[17], slots[17], positions[17];tokens[0]=0;
    for (int k = 0; k <= K; k++) { slots[k]=0;positions[k]=k+1;if(k)tokens[k]=drafts[k-1]; }
    rc = quinlan_spec_verify(d_out, target_logits, full_V, K+1, tokens, slots, positions, stream);
    if (!rc && cudaMemcpy(verified, d_out, (K+1)*sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) rc=-121;
    cudaFree(d_out);
    return rc;
}
