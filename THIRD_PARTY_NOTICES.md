# Third-party notices

## Qwentin

Bundled at `src/quinlan/_vendor/qwentin/`, from
[Qwentin](https://github.com/kacper-daftcode/qwentin) revision
`8a626374912fe1e24649403423c1120ff58d8522`, copyright 2026 qwentin authors,
under the MIT license retained in that directory and the top-level LICENSE.

Includes the native inference engine, scheduler, converter tools, CPU tests,
configuration/format docs, codebooks, draft vocabulary and recipe assets. Upstream raw benchmark reports, workflows and deployment examples are
omitted. The bundled README is a source notice, not the upstream project README.

Quinlan's native GPU sampling/MTP probability verifier and scheduler fixes are
already applied. Upstream machine-specific tool defaults and recipe provenance
paths were generalized. `src/quinlan/tool_parser.py` retains the upstream parser.
The historical native patch is kept for review and compatibility tooling.

## ExLlamaV3

Bundled at `src/quinlan/_vendor/exllamav3/`, from
[ExLlamaV3](https://github.com/turboderp-org/exllamav3) revision
`da68d57de61781e5b74ad68698ac4caf36408d7a`, copyright 2025 Turboderp,
under its MIT license, retained in that directory.

Includes library/conversion Python source, CUDA extension source, build metadata,
Hadamard tables. Calibration datasets are omitted for privacy. Unrelated benchmarks, examples,
research tools and tests are omitted. The converter has the diagnostic-reference
row environment override applied; user-supplied calibration data is unchanged. Its README is
a source notice. Two model-wrapper descriptions were shortened without changing
execution. Nested vendored components retain their own licenses, including
Flash Linear Attention and the ROCm WMMA kernels. See the LICENSE files under
`exllamav3/vendor/fla/` and `exllamav3/exllamav3_ext/rocm/vendor/rocm_wmma_gemm/`.

## Provenance and external dependencies

`src/quinlan/_vendor/SOURCES.json` records each included file's upstream SHA-256
where applicable and its integrated SHA-256. `quinlan-build --verify-only` checks
the integrated snapshot. No upstream Git history, model weights, credentials,
private deployment configs or conversation logs are distributed.

NVIDIA CUDA/CUB, PyTorch and other Python dependencies are installed separately
and retain their own licenses. This repository does not redistribute a CUDA
Toolkit, GPU driver or model weights. The API layer and router are adaptations
of the integration developed alongside these components, under the root MIT license.
