# Validation

The CPU suite covers HTTP auth/admission, sampling parameter propagation,
reasoning boundaries, nested tool arguments, streamed preamble/history equality,
JSON/SSE prefill fields, absence of TTFT substitution, scheduler timing hooks,
numeric-record bounds, byte-preserving routing, separate backend credentials,
sanitized backend errors, source pin checks, patch conflict handling, and parser
provenance. Real-tokenizer tests additionally exercise original-token replay
across repeated tool turns, equivalent API argument encodings, edited histories,
context-dependent templates, length-limited output, and bounded storage.

Set `QWENTIN_SOURCE` to a checkout containing the supported commit for overlay
checks. Set `QWENTIN_MODEL_DIR` to the matching local tokenizer directory for the
real-tokenizer tests. Those tests need tokenizer/config/template files only;
they do not load model weights or initialize CUDA. Missing optional inputs cause
explicit skips, not successful inference claims.

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
