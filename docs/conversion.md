# Included conversion tools

Quinlan contains the Qwentin exporter and ExLlamaV3 quantizer/library source.
There is no extra Git checkout or separately patched runtime to prepare. Supply
your own pinned source weights and calibration tokens and sufficient GPU/host memory and temporary disk.

| Component | Upstream revision |
| --- | --- |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) weights, external | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| [Qwentin](https://github.com/kacper-daftcode/qwentin), included | `8a626374912fe1e24649403423c1120ff58d8522` |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3), included | `da68d57de61781e5b74ad68698ac4caf36408d7a` |

Opt-5.75 is a target bitrate recipe with mixed 4–7-bit linear weights, a 6 bpw
output head, 4 bpw MTP weights and BF16 embeddings. The serving profile keeps
weights/embeddings on the GPU, uses trellis4 KV and dstate=4, and disables host parking.

```sh
# Use an isolated environment with the inference dependencies already installed.
python -m pip install '.[conversion]'

# Optional: retain one diagnostic output-logit reference row instead of five.
# This does not reduce calibration data or change the bitrate recipe.
export EXL3_CONVERT_DIAGNOSTIC_ROWS=1

# Set each variable to the appropriate source, temporary-work or output path.
quinlan-convert quantize \
  -i "$SOURCE_MODEL" -w "$CONVERSION_WORK" -o "$EXL3_EXPORT" \
  --recipe opt5.75 -cd "$CALIBRATION_TOKENS" -hb 6 -mb 4 -vb 16 -d 0 -cpi 300
quinlan-convert draft "$EXL3_EXPORT" -o "$DRAFT_HEAD"
quinlan-convert export "$EXL3_EXPORT" -o "$MODEL_TQF" --mtp-exl3 --draft-head "$DRAFT_HEAD"
quinlan-convert inspect "$MODEL_TQF" --expect exl3-mtp --verify
```

`--recipe opt5.75` resolves to the recipe inside the package. A custom recipe path
also works. The conversion wrapper puts the included ExLlamaV3 source first on
its Python path; its CUDA extension is compiled with the local PyTorch/toolkit
when needed. No upstream Git fetch occurs. Building that extension can take time;
it does not change the native Qwentin engine or an existing service.

Codebooks and draft vocabulary are retained with source hashes. Calibration
datasets are excluded because upstream corpora can contain personal contact
information. Provide `-cd/--cal_data` as a safetensors file of packed token rows
for the matching tokenizer. Calibration choice affects quantization quality;
use representative data and validate the result. No model weights or calibration
datasets are provided or downloaded by these commands.
Source weight/tokenizer licensing remains the publisher's responsibility and
the operator must follow it.

The initial export was 21,512,656,564 bytes, including 21,512,648,192 device weight
bytes, with SHA-256
`c3362ba9b9776cd48d40962452d5bd8c46436b8cc9cfde7819ba4568297ab1a5`.
That identifies the development artifact; GPU/toolchain changes can affect
byte-identical conversion. Different exports or profile settings need fresh
memory, speed and quality checks.
