# API, caching, and prefill measurements

## Supported requests

`POST /v1/chat/completions` supports `messages`, `model`, `stream`,
`stream_options.include_usage`, `max_tokens` (or `max_completion_tokens`),
`tools`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `reasoning_effort`,
and `chat_template_kwargs`. `tool_choice: "none"` disables tools; forced tool
selection and strict structured output are not implemented. The model still
determines whether and how it calls a tool. With `--vision-model-dir`, user/tool
content arrays may include base64 `image_url` parts; see [vision](vision.md).

Defaults: temperature 1, top-p 0.95, top-k 20, min-p 0.05, random 64-bit seed,
and xhigh thinking. Reasoning aliases: high/max → xhigh, minimal → low,
off → none. `chat_template_kwargs` can override thinking and
`preserve_thinking`; preserving thinking defaults to true. Developer messages
are rendered as system messages. JSON tool-argument strings are decoded for
the Qwen template, including nested objects.

Other OpenAI fields are not generally implemented or validated. In particular,
do not assume support for `n`, custom stop sequences, penalties, logprobs,
response schemas, video/audio, or network image URLs.

## Cache repair

Parsing raw model output into `reasoning_content`, `content`, and `tool_calls`
loses original whitespace and tool syntax. Re-rendering those fields can produce
different tokens and break the engine's exact prefix match. The repair keeps
original input/output token IDs and matches a continuation against the complete
API message history, tools, and rendering options.

Equivalent JSON argument encodings/key order and JavaScript numeric serialization
can match without changing the original inference tokens. Literal meaningful text,
reasoning, call IDs, tool-result IDs, tools, or template-option edits invalidate
the match. A template-prefix guard rejects context-dependent history rewrites,
including removed old thinking. Only the new canonical suffix is tokenized and
appended to the original tokens.

The final assistant content is emitted before tool-call SSE deltas. The same
assistant projection is remembered before terminal events, so an immediate tool
continuation finds its entry. The native scheduler also rejects reusing an
exactly consumed prompt with a stale pending output seed.

`TokenHistory` keeps at most eight entries and 32 MiB of logical token/template
payload. Python object overhead is additional. It is process-local and stores
no GPU/KV snapshots. `x_qwentin.token_history_preserved_tokens` describes a
token-history match; **actual GPU reuse** is
`usage.prompt_tokens_details.cached_tokens` / `x_qwentin.prefix_cached_tokens`.
A history hit is not a promise that the GPU prefix still exists.

## Pi prefill compatibility

The JSON response, final SSE choice event, and optional usage SSE event include:

```json
{
  "timings": {
    "prompt_n": 100,
    "prompt_ms": 50.0,
    "prompt_per_second": 2000.0,
    "cache_n": 9900,
    "scope": "server_prefill_wall"
  }
}
```

`prompt_n = prompt_tokens - cached_tokens`. PP is processed tokens divided by
measured prefill seconds. This is wall time from the scheduler beginning prefill
until its first output seed is ready. It includes waits between prefill waves
and seed selection, and excludes prior admission queueing, HTTP transport,
tokenization, client/tool time, and subsequent decode. No extra GPU synchronization
or per-token timing is added. The qualified profile disables host KV restoration.

TTFT remains separate in `x_qwentin.server_ttft_s`. Missing, failed, zero-work,
or invalid prefill measurements omit `timings`; TTFT is never substituted for PP.
Image encoding/upload occurs before native prefill and is excluded from PP;
`x_qwentin.vision.prepare_ms` reports that separate phase. It includes embedding
cache lookup/upload and native position configuration, not HTTP image decoding
or CPU preprocessing. Text TTFT retains its existing scheduler scope.
The final choice event carries timings even without `include_usage`. Pi needs a
timing extension that reads these fields to show PP; it can always receive the
OpenAI-style cache usage fields when streaming usage is enabled.

## Sampling and speculative decoding

Filtering runs on-device in this order: temperature → top-k → top-p (include the
crossing token) → min-p. Top-k 0 or -1 disables that filter. Seed and output
position select random draws independently of batch partitioning.

Probability mode samples from the draft distribution q, accepts proposals with
`min(1, p/q)`, and samples a rejection correction from normalized `max(p-q, 0)`.
It supports the reduced-vocabulary draft head by mapping q into the target
vocabulary. Deterministic mode verifies target-sampled tokens against proposals;
its acceptance/speed can differ. Greedy temperature-zero decoding retains the
upstream path. This package does not provide a newly trained draft model.
