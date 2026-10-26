# Validation

The CPU suite covers HTTP auth/admission, sampling parameter propagation,
reasoning boundaries, nested tool arguments, streamed preamble/history equality,
JSON/SSE prefill fields, absence of TTFT substitution, scheduler timing hooks,
numeric-record bounds, byte-preserving routing, separate backend credentials,
sanitized backend errors, source pin checks, patch conflict handling, and parser
provenance. Real-tokenizer tests additionally exercise original-token replay
across repeated tool turns, equivalent API argument encodings, edited histories,
context-dependent templates, length-limited output, and bounded storage.

All engine/build/overlay inputs come from the included source tree. No upstream
checkout, Git metadata, or network access is required for these checks. The
legacy external-checkout patch utility is tested against source reconstructed
from the bundled snapshot. `quinlan-build --verify-only` checks every included
source/asset against its manifest.

Set `QWENTIN_MODEL_DIR` to matching local tokenizer/config/template files for the
real-tokenizer tests. They do not load model weights or initialize CUDA. Missing
that optional input produces explicit skips. CI tests the installed package and
the bundled engine's CPU tests on Python 3.10, 3.12 and 3.14.

To include native ABI/configuration and TQF integrity checks after a build:

```sh
QWENTIN_LIB="$(python -c 'from quinlan.bundled import library_path; print(library_path())')" \
  python -m pytest src/quinlan/_vendor/qwentin/tests -q
```

The source archive and wheel are checked for complete native/conversion assets,
nested licenses, matching hashes and absence of generated binaries/bytecode.
Build output is separate from bundled sources, so an installed wheel can build
without writing into site-packages.

## Version 0.2.0 checks

- 52 API/router/build/cache tests passed with the real tokenizer available.
- 84 bundled-engine tests passed against a CUDA 13.1 build on an RTX 5090.
  Without a compiled library, the CPU-only run passed 77 and explicitly skipped seven.
- The device sampler matched CPU oracles across seven filtering regimes and five
  probability-verifier cases, including two reduced-vocabulary pipeline cases,
  greedy preservation and batch-partition reproducibility.
- The source archive and installed wheel include every manifest file, nested
  licenses and required codebook/vocabulary asset, with calibration datasets,
  generated binaries and bytecode excluded.

These checks qualify packaging, native ABI/build and the sampling algorithm.
They do not constitute a new full-model quality or long-context performance benchmark.

## Version 0.3.0 image checks

- 61 API/router/build/cache/media CPU tests passed with the real tokenizer.
- 84 bundled-engine tests passed against the CUDA 13.1 build on an RTX 5090.
- The independent Transformers oracle matched three image-grid cases, all 32
  rotary frequency selections, device visual rows, generated-token positional
  deltas, and the return to ordinary text positions.
- 18 direct API checks and 19 routed checks passed, including color, spatial
  layout, OCR, multiple images, a 2048-square image with 4,096 visual tokens,
  queued image requests, tools, SSE metadata, and seeded stochastic output.
- An image/tool follow-up reused all 384 consumed tokens; appending an image
  reused all 427 eligible tokens. Replacing the old image produced the updated
  answer with zero cached tokens.
- The installed Pi provider completed an image/tool/answer round trip through
  a separate router, preserved its full consumed prefix, and its installed
  prefill parser detected the native PP fields.
- A 262,080-token image-containing prompt answered correctly through the router,
  with a configured 262,144-token limit. Prefill took approximately 160 seconds.
- Two deterministic text smoke fixtures returned identical messages before and
  after the change. Measured decode rates were approximately 131→131 and
  122→122 tokens/s. These single comparisons are not a controlled speed estimate.

[Sanitized numeric receipt](evidence/vision-0.3.0.json) records the configuration,
cases and limitations. The fixtures are synthetic and contain no personal data.
These checks establish the measured integration paths, not universal image
accuracy or reasoning quality throughout 256K. Video/audio and URL fetching are
not supported. Interleaved conversations can still evict the single parked GPU
prefix; a CPU image-embedding hit is not a KV-cache hit.

## GPU sampler oracle

With CUDA Toolkit 13.1 and CUDA PyTorch on the RTX 5090:

```sh
nvcc -O3 -std=c++17 -arch=sm_120 -shared -Xcompiler=-fPIC \
  -I src/quinlan/native tests/sampling_harness.cu -o /tmp/quinlan-sampling-test.so
python tests/check_sampling_gpu.py /tmp/quinlan-sampling-test.so
```

The harness compiles the same native sampler used by the engine without loading
model weights. It compares GPU output to CPU oracles across filter combinations,
distribution checks, ties/extreme logits, the full target vocabulary, seeded
reproducibility and batch partitions. It also checks acceptance/residual sampling
for several p/q pairs, deterministic proposal validation, and reduced-vocabulary
draft-to-target mapping. It is separate from default CPU discovery.

## Inference qualification

Before promoting a new engine build, export, or profile:

1. Inspect authenticated model discovery and effective `/health` configuration.
2. Generate a short response at temperature zero and with seeded sampling.
3. Complete several streamed tool-call/result continuations. Compare actual
   `cached_tokens` against the consumed prefix, not just token-history hits.
4. Check that nonempty tool preambles survive the streamed response and replay.
5. Verify `timings.prompt_n` counts uncached work, and timings survive your router
   and Pi timing extension with and without streaming-usage opt-in.
6. Measure long-context memory, prefill, decode speed, and task quality separately.

During development, the serving integration completed an on-device tool sequence
through a separate router and Pi timing extension: all six tested continuations
retained full eligible cache reuse. A cold prefill processed 15,064 tokens in
5.20 seconds (about 2,895 tok/s); a warm continuation processed 62 new tokens in
about 0.03 seconds. These are rounded observations for one workload/profile,
not reproducible performance guarantees or proof of 256K reasoning quality.
Raw deployment logs and conversations are intentionally not distributed.
