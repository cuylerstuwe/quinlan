# Quinlan

An OpenAI-style chat API for [Qwentin](https://github.com/kacper-daftcode/qwentin),
with exact token-history preservation, Pi-compatible prefill measurements, and
GPU sampling with speculative decoding. Includes an optional model-alias router.

The initial profile targets **Qwen3.8 27B, EXL3 opt-5.75, on a 32 GB RTX 5090**:
262,144 total context tokens, up to 32,768 output tokens, GPU-resident BF16
embeddings, trellis4 KV, and one active sequence. Capacity depends on the exported
weights and runtime configuration. A 256K capacity does not establish equal
reasoning quality across that whole context.

## Included

- Authenticated `/v1/models` and `/v1/chat/completions`, JSON or SSE, separated
  reasoning, parsed function calls, and optional streaming usage.
- Prompt-cache repair across assistant/tool round trips. Original generated token
  IDs survive JSON argument reformatting, reasoning extraction, and tool parsing.
  Streamed assistant preambles match the history retained for the next request.
- Pi timing fields: processed tokens, measured prefill wall time, PP tokens/sec,
  and actual cached tokens. TTFT is reported separately.
- Native temperature, top-k, top-p, min-p, seed handling, and probability-based
  MTP acceptance/rejection with residual sampling. Temperature zero keeps the
  upstream greedy path.
- A portable alias router that changes only the request's model ID and relays
  response bytes, including all timing and usage extensions, unchanged.
- Tests, pinned upstream overlay, setup instructions, and example Pi/router configs.

## Install and build

Serving requires Linux, an RTX 5090, a suitable NVIDIA driver, CUDA Toolkit 13.1,
CMake, Git, and a Python environment with Qwentin's pinned runtime dependencies.
CPU tests and the optional router work without a GPU. Python 3.10+ is supported
by this package; the inference profile was exercised on Python 3.12.

```sh
git clone https://github.com/cuylerstuwe/quinlan.git
cd quinlan
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .

git clone https://github.com/kacper-daftcode/qwentin.git vendor/qwentin
git -C vendor/qwentin checkout --detach 8a626374912fe1e24649403423c1120ff58d8522
python -m pip install 'torch==2.14.0+cu130' --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r vendor/qwentin/requirements.txt
quinlan-patch vendor/qwentin

# Set CUDA_NVCC to the nvcc executable from your CUDA 13.1 toolkit.
cmake -S vendor/qwentin -B vendor/qwentin/build-qwen \
  -DCMAKE_CUDA_COMPILER="$CUDA_NVCC" -DCMAKE_BUILD_TYPE=Release
cmake --build vendor/qwentin/build-qwen --parallel 2
```

The overlay checks the exact source revision and preflights all patch hunks.
Reapplying it is harmless. Rebuild after applying or updating it. An existing
unpatched Qwentin library lacks the `quinlan_*` sampling ABI and will not work.
Use a separate checkout/environment when another service depends on Qwentin.

## Serve

Supply a compatible exported TQF file and the corresponding local tokenizer/model
directory. Weights are not included. See [conversion](docs/conversion.md) for the
tested recipe and source pins.

```sh
mkdir -p secrets
chmod 700 secrets
python -c 'import pathlib,secrets; p=pathlib.Path("secrets/backend.key"); p.touch(mode=0o600,exist_ok=False); p.write_text(secrets.token_urlsafe(32)+"\n")'

# Set MODEL_DIR to the local model/tokenizer directory and MODEL_TQF to its TQF.
quinlan-serve \
  --source vendor/qwentin \
  --model-dir "$MODEL_DIR" \
  --tqf "$MODEL_TQF" \
  --api-key-file secrets/backend.key
```

The backend listens on `127.0.0.1:8000`. `--model-id`, `--host`, `--port`,
`--context-window`, `--max-output`, `--num-blocks`, `--no-spec`, `--spec-k`,
`--draft-vocab`, and `--spec-sampling` are configurable; see `quinlan-serve --help`.
The default pool is 2,200 blocks of 128 tokens. Lower contexts must be multiples
of 128. Startup verifies effective context, embedding placement, state precision,
KV format, and disabled host parking.

`GET /health` on the backend is unauthenticated and exposes numeric counters and
configuration, not prompts. Keep it on a trusted interface. Both services use
plain HTTP; use your own TLS proxy or private tunnel for remote access. Neither
service configures a firewall, certificates, system services, or public ingress.

## Optional router and Pi

The router can run on the inference host or another trusted machine. Its
configuration names environment variables instead of containing credentials.

```sh
export QUINLAN_BACKEND_KEY="$(cat secrets/backend.key)"
export QUINLAN_ROUTER_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
quinlan-router examples/router.json
```

Change the example's backend URL if it runs elsewhere. Its `/v1/models` exposes
the `qwen-local` alias. Chat response `model` fields retain the backend's model ID;
the response body/SSE is passed through unchanged. The router does not retry,
fail over, or reformat conversation history. It is a standalone extraction for
this integration; deployment-specific scheduling and native pause/resume
controllers are outside this package.

[examples/pi-provider.json](examples/pi-provider.json) is a provider fragment for
Pi's `models.json`. Merge it into your existing configuration, and make
`QUINLAN_ROUTER_KEY` available to Pi. For a remote router, update the base URL.
Use a context window and output limit that match the server. Pi's usage accounting
receives `usage.prompt_tokens_details.cached_tokens`; a timing extension that
understands llama.cpp-style `timings` can display measured PP. The timing
extension itself is not bundled.

See [API and compatibility details](docs/compatibility.md) for fields, cache
semantics, and supported behavior.

## Test

```sh
python -m unittest discover -s tests -p 'test_*.py' -v

# Also verify patch application, conflict handling and parser provenance:
QWENTIN_SOURCE="$PWD/vendor/qwentin" \
  python -m unittest discover -s tests -p 'test_*.py' -v

# Also run the real-tokenizer cache regressions (no inference or GPU required):
python -m pip install '.[tokenizer]'
QWENTIN_MODEL_DIR="$MODEL_DIR" QWENTIN_SOURCE="$PWD/vendor/qwentin" \
  python -m unittest discover -s tests -p 'test_*.py' -v
```

GPU oracle tests are opt-in; see [validation](docs/validation.md). CI runs CPU HTTP,
telemetry, router, and pinned-overlay checks. It does not download model weights or
claim GPU qualification.

## Scope and privacy

This is an experimental, pinned integration, not a complete OpenAI API server.
It serves text chat and tool calls, one active sequence, with no vision,
embeddings endpoint, persistent conversation storage, native priority scheduling,
or enforced JSON grammar. It retains bounded token/template history in RAM and
the engine's GPU prefix state. Restarting, eviction, interleaving conversations,
or editing history can legitimately cause a cold prefill.

The repository contains portable source and synthetic fixtures. Credentials,
private network addresses, hostnames, personal paths, conversation logs, raw
deployment evidence, model weights, binaries, and private deployment configuration
are excluded. The services avoid logging request text, tool arguments, and
authorization headers. `.gitignore` provides a guardrail; review staged files
before publishing your own configuration or evidence.

MIT license; see [third-party notices](THIRD_PARTY_NOTICES.md).
