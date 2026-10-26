// Called only on the engine owner between requests. Embeddings are borrowed
// device BF16 memory kept alive by the Python owner; positions are copied.
extern "C" int quinlan_vision_configure(const void *embeddings, int rows,
                                        const int *positions, int length, int delta) {
    if (!g_pg_ready || g_pg_maxslots != 1) return -1;
    if (rows < 0 || rows > 65536 || length < 0 || length > g_qwen.max_seq) return -2;
    if (rows && (!embeddings || !positions || !length || !g_qwen.mrope_interleaved)) return -3;
    if (cudaStreamSynchronize(g_qwen.stream) != cudaSuccess) return -4;
    quinlan_visual_state state = {};
    if (rows) {
        for (int i = 0; i < length * 3; ++i)
            if (positions[i] < 0 || positions[i] >= g_qwen.max_seq) return -5;
        if (length > qv_capacity) {
            graph_invalidate();
            if (qv_positions) cudaFree(qv_positions);
            qv_positions = NULL; qv_capacity = 0;
            if (cudaMalloc(&qv_positions, (size_t)length * 3 * sizeof(int)) != cudaSuccess) return -6;
            qv_capacity = length;
        }
        if (cudaMemcpy(qv_positions, positions, (size_t)length * 3 * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return -7;
        state = {(const uint16_t *)embeddings, qv_positions, rows, length, delta,
                 g_qwen.mrope_section[1], g_qwen.mrope_section[2]};
    }
    return cudaMemcpyToSymbol(qv_state, &state, sizeof(state)) == cudaSuccess ? 0 : -8;
}

// Small independent device oracle for all 32 rotary frequencies and visual rows.
__global__ void quinlan_vision_probe_kernel(int *positions_out, float *embeddings_out,
                                           int start, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count * 32) positions_out[i] = qv_rope_position(start + i / 32, i % 32);
    if (i < qv_state.rows && embeddings_out)
        embeddings_out[i] = qv_embedding((size_t)(-i - 1), 0, 5120);
}
extern "C" int quinlan_vision_probe(int start, int count, int *out, float *embedding_out) {
    if (start < 0 || count < 1 || count > 1024) return -1;
    int *d = NULL; float *e = NULL;
    if (cudaMalloc(&d, (size_t)count * 32 * sizeof(int)) != cudaSuccess) return -2;
    if (cudaMalloc(&e, 65536 * sizeof(float)) != cudaSuccess) { cudaFree(d); return -2; }
    quinlan_vision_probe_kernel<<<(count * 32 + 255) / 256, 256>>>(d, e, start, count);
    int rc = cudaMemcpy(out, d, (size_t)count * 32 * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : -3;
    if (embedding_out && cudaMemcpy(embedding_out, e, count * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) rc = -3;
    cudaFree(d); cudaFree(e); return rc;
}
