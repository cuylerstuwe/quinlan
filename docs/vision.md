# Image understanding

Quinlan can attach Qwen's BF16 vision tower to its existing EXL3 text/MTP decoder.
The text weights, KV format, sampling, speculative decoding, and 262,144-token
context configuration remain the same. Images occupy positions within that
context budget. This release supports still images; video/audio are not exposed.

Install and launch:

```sh
python -m pip install '.[inference,vision]' --extra-index-url https://download.pytorch.org/whl/cu130
quinlan-build
quinlan-serve --model-dir "$MODEL_DIR" --tqf "$MODEL_TQF" \
  --vision-model-dir "$VISION_MODEL_DIR" --api-key-file secrets/backend.key
```

`VISION_MODEL_DIR` must hold matching BF16 `model.visual.*` tensors, a vision
configuration matching the text checkpoint, and `preprocessor_config.json`.
It can be the original model checkpoint or a compatible EXL3 checkpoint that
retains the tower unquantized. Use `--vision_bits 16` (`-vb 16`) when converting
with the included ExLlama quantizer. Quantized vision tensors are not supported
by this bridge. Model weights are not distributed in this repository.

The TQF exporter still emits the text/MTP artifact separately. Keep the vision
checkpoint available; deleting it disables vision startup. Supplying only a
text TQF cannot reconstruct missing vision weights.

## API and clients

Image attachments use OpenAI-style base64 data URLs:

```python
import base64, json, os, urllib.request
from pathlib import Path

body = {
    "model": "qwen-local",  # router alias, or the backend ID for direct requests
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(Path("image.png").read_bytes()).decode()
        }}
    ]}],
    "max_tokens": 256,
    "reasoning_effort": "none"
}
request = urllib.request.Request(
    "http://127.0.0.1:8787/v1/chat/completions", data=json.dumps(body).encode(),
    headers={"Authorization": "Bearer " + os.environ["QUINLAN_ROUTER_KEY"],
             "Content-Type": "application/json"})
with urllib.request.urlopen(request) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
```

JSON and streaming responses, function tools, and images in tool-result content
are supported. Pi's model declaration needs `"input": ["text", "image"]`; the
example provider and router declarations include this. Backend model discovery
advertises images only when a tower is loaded. The router forwards image bodies
and timing/cache fields without a separate model or endpoint.

The service deliberately does not fetch HTTP(S), file, or other image URLs.
Clients should download their own authorized images and attach their bytes.
This avoids giving arbitrary requests access to the server's network or files.
Pillow decodes the image format; MIME headers do not select a filesystem path.

Default bounds: eight images, 20 MiB of encoded-image payload bytes per image,
32 Mi pixels in a source image, 4,194,304 processed pixels per image, and 8,192
total visual tokens per request. The source-image limit applies before pixel
decoding. EXIF orientation is honored, images are converted to RGB, and animated
images are rejected. Processor rounding/merging determines exact token counts.
The API's `detail` hint does not currently change the configured processing bounds.
Both server and router also enforce their independent HTTP body-size limits.

## Native positions and cache correctness

The Python processor expands image placeholders before context admission. On
the owner thread, the encoder produces 5,120-dimensional BF16 visual embeddings;
negative internal IDs select these rows in the native embedding lookup. They
are never exposed as model vocabulary IDs. Qwen's interleaved temporal/height/
width rotary coordinates are applied to both query and key paths, including
prefill, decode, and the MTP drafter. Causal/KV indices remain ordinary sequence
positions. Generated tokens use the image-adjusted positional delta.

This bridge requires one native slot. HTTP requests may queue, but the encoder
and decoder GPU operations run on the engine owner, so queued requests cannot
replace the active request's image state.

Prompt-history keys include image content digests. The original-token repair
works with expanded image placeholders and the returned assistant/tool fields.
Unchanged image histories can reuse their full consumed prefix. Appending new
images retains previous visual-row identities and eligible cached history.
Replacing, removing, or reordering an earlier image invalidates the GPU prefix
before matching; repeated placeholder IDs alone are insufficient evidence.

A process-local CPU BF16 embedding cache has a 128 MiB logical tensor budget.
It stores image digests, grids and embeddings, not credentials or persisted
conversations. Python object overhead and queued-request references are extra.
The loaded tower and current visual embeddings stay on GPU. CPU embedding
caching is independent of GPU prompt/KV reuse; a hit does not mean the decoder
has retained the conversation's prefix.
Interleaving another conversation can evict the single parked GPU prefix even
when its images remain in the CPU embedding cache.

Text-only requests bypass the encoder and do not add a native vision call when
already in text mode. Returning from an image request clears the native visual
state before text decoding. Prefill telemetry remains measured decoder work;
image preparation has its own numeric extension and is excluded from PP.

## Qualification

The optional `tests/check_vision_live.py` exercises synthetic color, layout, OCR,
multi-image ordering, image/tool continuations, changed/appended images, queued
images, stochastic repeatability, SSE usage/timings, and text after images.
Run it once directly and again against the router alias:

```sh
python tests/check_vision_live.py --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.8-27b-qwentin-opt5.75-256k --key-file secrets/backend.key \
  --output /tmp/vision-check.json
```

`tests/check_vision_native.py` independently compares image positions and all 32
rotary frequency selections with the pinned Transformers implementation, checks
visual embedding rows on device, and verifies text-state reset. See
[validation](validation.md) for measured results and limits. Passing simple
fixtures does not establish universal image accuracy or long-context reasoning.
