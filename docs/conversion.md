# Model export and profile

Quinlan serves a Qwentin TQF export. It does not read arbitrary GGUF files or raw
Hugging Face shards. Obtain source weights from their publisher or supply your
own compatible export. Conversion is optional when you already have that export.

The initial profile uses these public sources:

| Component | Revision |
| --- | --- |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| [Qwentin](https://github.com/kacper-daftcode/qwentin) | `8a626374912fe1e24649403423c1120ff58d8522` |
| [ExLlamaV3](https://github.com/turboderp-org/exllamav3) | `da68d57de61781e5b74ad68698ac4caf36408d7a` |

Opt-5.75 is a **target bitrate recipe**, not uniform 5.75-bit tensors. It mixes
4–7-bit linear weights; the qualified export uses a 6 bpw output head, 4 bpw MTP
weights, and BF16 embeddings. The serving profile keeps embeddings and weights
on the GPU. It uses trellis4 KV and dstate=4 with host parking disabled.

After installing the pinned Qwentin runtime, build/install ExLlamaV3 in the same
isolated environment according to its requirements. Its extension build needs
the CUDA/PyTorch toolchain. These commands do not install a driver or toolkit
and do not download source weights automatically.

```sh
git clone https://github.com/turboderp-org/exllamav3.git vendor/exllamav3
git -C vendor/exllamav3 checkout --detach da68d57de61781e5b74ad68698ac4caf36408d7a
python -m pip install setuptools wheel ninja packaging pyyaml
python -m pip install --no-build-isolation -e vendor/exllamav3

# Set these to separate source, temporary-work, and output directories.
# SOURCE_MODEL must contain the complete pinned source weights and tokenizer.
python vendor/exllamav3/convert.py \
  -i "$SOURCE_MODEL" -w "$CONVERSION_WORK" -o "$EXL3_EXPORT" \
  -rcp vendor/qwentin/tools/recipes/qwen3_8-27b-exl3-opt5.75.yaml \
  -hb 6 -mb 4 -vb 16 -d 0 -cpi 300

python vendor/qwentin/tools/draft_head_build.py "$EXL3_EXPORT" -o "$DRAFT_HEAD"
python vendor/qwentin/tools/convert_qwen_tqf.py "$EXL3_EXPORT" \
  -o "$MODEL_TQF" --mtp-exl3 --draft-head "$DRAFT_HEAD"
python vendor/qwentin/tools/inspect_tqf.py "$MODEL_TQF" --expect exl3-mtp --verify
```

The initial export was 21,512,656,564 bytes with 21,512,648,192 device weight bytes.
Its SHA-256 was
`c3362ba9b9776cd48d40962452d5bd8c46436b8cc9cfde7819ba4568297ab1a5`.
This identifies that artifact; it is not a promise of byte-identical conversion
across GPU/toolchain versions. No weights are hosted by Quinlan.

Conversion needs substantial host RAM and temporary disk space in addition to
GPU memory. A diagnostic-logit retention workaround used during development is
provided as [limit_conversion_diagnostics.py](../scripts/limit_conversion_diagnostics.py).
It applies only to the pinned ExLlamaV3 revision. Setting
`EXL3_CONVERT_DIAGNOSTIC_ROWS=1` after applying it reduces retained output-logit
reference rows; it does not reduce calibration data. The default remains five
reference rows. This is separate from the quantization recipe.

The 262,144-token capacity is shared by prompt and requested output. The server
rejects requests whose sum exceeds it; a 256K prompt plus a 32K answer will not
fit. Keep client context declarations aligned with the server and leave an
output budget. Full-vocabulary drafting, different pool sizes, weights, or
precision settings require fresh memory/performance qualification.
