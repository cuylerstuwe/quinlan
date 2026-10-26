// Optional single-slot visual embeddings and Qwen interleaved three-axis RoPE.
// The decoder's causal/KV positions remain ordinary sequence positions.
struct quinlan_visual_state {
    const uint16_t *embeddings;
    const int *positions; // [prompt positions][temporal, height, width]
    int rows, length, delta, section_h, section_w;
};
static __constant__ quinlan_visual_state qv_state = {};
static int *qv_positions = NULL;
static int qv_capacity = 0;

static __device__ __forceinline__ int qv_rope_position(int pos, int frequency_index) {
    if (!qv_state.positions) return pos;
    if (pos >= qv_state.length) return pos + qv_state.delta;
    if (pos < 0) return pos;
    int axis = 0;
    if (frequency_index % 3 == 1 && frequency_index < 3 * qv_state.section_h) axis = 1;
    if (frequency_index % 3 == 2 && frequency_index < 3 * qv_state.section_w) axis = 2;
    return qv_state.positions[(size_t)pos * 3 + axis];
}

// Negative internal IDs index request-local visual rows. API/tokenizer IDs
// stay unchanged until the final native-input projection.
static __device__ __forceinline__ float qv_embedding(size_t row, int channel, int hidden) {
    size_t visual_row = (size_t)(-(int64_t)row - 1);
    if (!qv_state.embeddings || visual_row >= (size_t)qv_state.rows) return 0.0f;
    return tq_bf16_to_float(qv_state.embeddings[visual_row * hidden + channel]);
}
