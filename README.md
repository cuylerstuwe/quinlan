# Quinlan

A complete Qwentin serving stack in one repository: CUDA inference engine,
continuous-batching scheduler, OpenAI-style chat API, model-alias router,
exact token-history preservation, optional BF16 image understanding, Pi prefill
telemetry, and conversion tools.

Qwentin and ExLlamaV3 source is checked in as regular files. Building, serving,
routing, and conversion require no submodules, external source checkouts, or
patch step. The source archive and Python wheel include the same source assets.
CUDA/PyTorch, other Python packages, system build tools, and model weights/tokenizer
files and conversion calibration data remain external requirements.

The initial profile targets **Qwen3.8 27B, EXL3 opt-5.75, on a 32 GB RTX 5090**:
262,144 total context tokens, up to 32,768 output tokens, GPU-resident BF16
embeddings, trellis4 KV, and one active sequence. Actual capacity depends on the
export and runtime configuration. A 256K capacity does not establish equal
reasoning quality throughout that context.

## Included

- CUDA inference source, C ABI, Python scheduler, TQF loader/exporter, codebooks,
  reduced draft vocabulary, quantization recipes, and conversion source.
- Temperature, top-k, top-p, min-p, seeded sampling, and probability-based MTP
  acceptance/rejection with residual sampling. Temperature zero retains greedy decoding.
- Authenticated `/v1/models` and `/v1/chat/completions`, JSON or SSE, separated
  reasoning, parsed function calls, and optional streaming usage.
- Prompt-cache repair across assistant/tool round trips, including streamed
  preambles, reasoning extraction, JSON argument reformatting, and tool parsing.
- Pi-compatible processed-token counts, measured prefill wall time, PP tokens/sec,
  and actual cached tokens. TTFT is reported separately.
- Optional Qwen image encoder, native visual-embedding input and three-axis RoPE,
  with image-aware prompt caching and a bounded CPU embedding cache.
- An authenticated model-alias router that translates the request model ID and
  passes response bytes, including timing/cache/usage extensions, unchanged.
- Tests, example Pi/router configs, upstream licenses, revisions, and file hashes.

## Install and build

Inference requires Linux, an RTX 5090, a suitable NVIDIA driver, CUDA Toolkit 13.1,
CMake, a C++ compiler, and Python 3.10+. The inference profile was exercised on
Python 3.12. CPU tests and the router also run without a GPU.

```sh
git clone https://github.com/cuylerstuwe/quinlan.git
cd quinlan
python3 -m venv .venv
. .venv/bin/activate

# Explicit cu130 PyTorch pin; use the CUDA wheel index as well as PyPI.
python -m pip install '.[inference]' --extra-index-url https://download.pytorch.org/whl/cu130

# Point CUDACXX at nvcc from your CUDA 13.1 toolkit if it is not on PATH.
quinlan-build
```

That is the only source checkout needed. An unpacked source archive works too;
Git is not used by the build or server. `quinlan-build` compiles the included
engine and prints the library path. By default, build output goes in a source-keyed
user cache, leaving the installed sources writable only when you choose to edit
them. Set `QUINLAN_BUILD_DIR` for a shared custom build directory. If using
`quinlan-build --build-dir build`, pass `--lib build/libforward_qwen.so` to the server.
The repository also supports `cmake -S . -B build` with the correct CUDA compiler.

```sh
# Verify all bundled source and asset hashes without CUDA:
quinlan-build --verify-only

# Router-only or CPU development installation:
python -m pip install -e '.[dev]'
```

The sampling and scheduler changes are already applied in the bundled engine.
`quinlan-patch` remains a compatibility utility for an old external Qwentin
checkout; normal Quinlan builds do not call it. The CUDA build contains no
FetchContent, ExternalProject, or source-download step.

## Serve

Supply a compatible exported TQF and its matching tokenizer/model directory.
Weights are not included. See [conversion](docs/conversion.md) for the included
conversion commands and exact source pins.

```sh
mkdir -p secrets
chmod 700 secrets
python -c 'import pathlib,secrets; p=pathlib.Path("secrets/backend.key"); p.touch(mode=0o600,exist_ok=False); p.write_text(secrets.token_urlsafe(32)+"\n")'

# Set MODEL_DIR and MODEL_TQF to your local tokenizer directory and export.
quinlan-serve --model-dir "$MODEL_DIR" --tqf "$MODEL_TQF" \
  --api-key-file secrets/backend.key
```

The server automatically selects the included scheduler and `quinlan-build` output.
It listens on `127.0.0.1:8000`. Model ID, bind address/port, context/output limits,
KV block count, speculation mode/depth and draft vocabulary are configurable;
see `quinlan-serve --help`. `--source` and `--lib` are optional advanced overrides.

The default KV pool is 2,200 blocks of 128 tokens. Lower contexts must be multiples
of 128. Startup verifies effective context, embedding placement, state precision,
KV format, and disabled host parking. Prompt plus requested output must fit the
context window.

To enable image attachments, install `.[inference,vision]` and add
`--vision-model-dir "$VISION_MODEL_DIR"`. This directory must contain matching
BF16 `model.visual.*` safetensors, `config.json`, and `preprocessor_config.json`.
The TQF remains the quantized text/MTP component; it does not contain the tower.
See [vision setup, API examples, limits, and cache behavior](docs/vision.md).

`GET /health` on the backend is unauthenticated and exposes numeric counters and
configuration, not prompts. Keep it on a trusted interface. Both services use
plain HTTP; provide your own TLS proxy or private tunnel for remote access.
The package does not configure system services, a firewall, or public ingress.

## Router and Pi

The router is part of this package and can run on the inference host or another
trusted machine. Its config names environment variables rather than credentials.

```sh
export QUINLAN_BACKEND_KEY="$(cat secrets/backend.key)"
export QUINLAN_ROUTER_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
quinlan-router examples/router.json
```

The example exposes `qwen-local` at `http://127.0.0.1:8787/v1` and advertises image
input, so run the backend with its vision option or change the examples to text
only. Change its backend
URL for a separate host. Response `model` fields retain the backend model ID;
JSON/SSE response bytes pass through unchanged. The router does not retry, fail
over, or reformat history. Deployment-specific priority controllers are outside
the supported profile.

[examples/pi-provider.json](examples/pi-provider.json) is a provider fragment for
Pi's `models.json`. Merge it into your existing config and make
`QUINLAN_ROUTER_KEY` available to Pi. Keep the context window/output limit aligned
with the server. Pi receives `usage.prompt_tokens_details.cached_tokens`; a
Pi timing extension that understands llama.cpp-style `timings` can display PP.
The extension itself is not required for inference and is not bundled.

See [API and cache details](docs/compatibility.md).

## Test and provenance

```sh
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -p 'test_*.py' -v
python -m pytest src/quinlan/_vendor/qwentin/tests -q

# Also exercise exact cache round trips with the real tokenizer; no GPU needed:
python -m pip install '.[tokenizer]'
QWENTIN_MODEL_DIR="$MODEL_DIR" python -m unittest discover -s tests -p 'test_*.py' -v
```

CI uses the included sources; it does not clone upstream repositories. GPU oracle
and native build checks are described in [validation](docs/validation.md).

Source layout:

| Location | Contents |
| --- | --- |
| `src/quinlan/` | API, cache/telemetry, router, build/conversion entry points |
| `src/quinlan/_vendor/qwentin/` | Inference engine, scheduler, TQF/conversion tools, tests/assets |
| `src/quinlan/_vendor/exllamav3/` | Quantizer/library source, CUDA extension source, runtime assets |
| `src/quinlan/_vendor/SOURCES.json` | Original revisions, upstream hashes and integrated-file hashes |
| `CMakeLists.txt` | Top-level build for the included engine |

The upstream commits and modifications are documented in
[third-party notices](THIRD_PARTY_NOTICES.md). Public upstream codebooks and draft vocabulary are included. Model weights,
calibration datasets and raw deployment logs are not distributed. Credentials, private addresses/hostnames/paths, and personal
configuration are excluded. Commit metadata uses a neutral contributor identity.

This is an experimental text/image chat and tool integration with one active
sequence. It has no video/audio support, embeddings endpoint, enforced JSON grammar, persistent
conversation storage, or native priority scheduling. Token/template history is
bounded in RAM. Restarting, eviction, interleaving conversations, or genuine
history edits can legitimately cause a cold prefill. MIT license; bundled
components retain their own notices and licenses.
