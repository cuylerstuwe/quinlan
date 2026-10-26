#!/usr/bin/env python3
"""Pi/OpenAI front end for Qwentin plus the isolated Quinlan sampling overlay.

GPU sampling, Qwen thinking/effort, live reasoning SSE, parsed tools,
streaming usage, authenticated model discovery, and context admission.
"""
import argparse
from array import array
from collections import OrderedDict
import copy
from dataclasses import dataclass
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import queue
import secrets
import sys
import threading
import time
import uuid

CONTEXT = 262144
MAX_OUTPUT = 32768
MODEL = "qwen3.8-27b-qwentin-opt5.75-256k"

@dataclass(frozen=True)
class ServerConfig:
    model: str = MODEL
    context_window: int = CONTEXT
    max_output: int = MAX_OUTPUT

def sampling_options(body):
    temperature = float(body.get("temperature", 1.0))
    top_p = float(body.get("top_p", 0.95))
    top_k = int(body.get("top_k", 20))
    if top_k == -1:
        top_k = 0
    min_p = float(body.get("min_p", 0.05))
    if not math.isfinite(temperature) or not 0 <= temperature <= 10:
        raise ValueError("temperature must be finite and between 0 and 10")
    if not math.isfinite(top_p) or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError("top_p must be in (0,1]; top_k must be -1, 0, or positive")
    if not math.isfinite(min_p) or not 0 <= min_p <= 1:
        raise ValueError("min_p must be in [0,1]")
    seed = body.get("seed")
    if seed is None:
        seed = secrets.randbits(64)
    elif isinstance(seed, bool) or not isinstance(seed, int) or not -(2**63) <= seed < 2**64:
        raise ValueError("seed must be a 64-bit integer")
    return temperature, top_p, top_k, min_p, seed % 2**64

def split_reasoning(text, thinking, final=False):
    """Split a growing decode without leaking a partial think delimiter."""
    if "<think>".startswith(text) and text and not final:
        return "", ""
    explicit = text.startswith("<think>")
    if explicit:
        text = text[len("<think>"):].lstrip("\n")
    if thinking or explicit:
        if "</think>" in text:
            reasoning, content = text.split("</think>", 1)
            return reasoning, content.lstrip("\n")
        if not final:
            marker = "</think>"
            for n in range(min(len(marker)-1, len(text)), 0, -1):
                if text.endswith(marker[:n]):
                    text = text[:-n]
                    break
        return text, ""
    return "", text

def template_options(body):
    opts = dict(body.get("chat_template_kwargs") or {})
    effort = opts.get("reasoning_effort", body.get("reasoning_effort", "xhigh"))
    if effort is None:
        effort = "none"
    effort = {"high": "xhigh", "max": "xhigh", "minimal": "low", "off": "none"}.get(str(effort).lower(), str(effort).lower())
    if effort not in {"none", "low", "medium", "xhigh"}:
        raise ValueError("reasoning_effort must be none, low, medium, or xhigh")
    thinking = bool(opts.get("enable_thinking", body.get("enable_thinking", effort != "none")))
    opts.update(enable_thinking=thinking, reasoning_effort=effort, preserve_thinking=opts.get("preserve_thinking", True))
    return opts

def prepare_messages(messages):
    out = copy.deepcopy(messages)
    for m in out:
        if m.get("role") == "developer":
            m["role"] = "system"
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if isinstance(fn.get("arguments"), str):
                fn["arguments"] = json.loads(fn["arguments"])
    return out

class PrefillTelemetry:
    """Wall time from scheduling prefill until its first output seed is ready.

    Queue wait before prefill, HTTP/tokenization and output decoding are excluded.
    Waits between prefill waves are included: this is wall time, not GPU compute.
    Only bounded numeric records are retained; no request/prompt references.
    """
    def __init__(self, clock_ns=time.perf_counter_ns, max_records=1024):
        self.clock_ns = clock_ns
        self.max_records = max_records
        self.records = OrderedDict()
        self.lock = threading.Lock()

    def begin(self, req):
        start = self.clock_ns()
        record = {"start_ns": start, "end_ns": None, "prompt": req.n_prompt, "cached": req.start}
        with self.lock:
            self.records.pop(id(req), None)
            self.records[id(req)] = record
            while len(self.records) > self.max_records:
                self.records.popitem(last=False)

    def finish(self, req):
        end = self.clock_ns()
        with self.lock:
            record = self.records.get(id(req))
            if record is not None:
                record["end_ns"] = end

    def discard(self, req):
        with self.lock:
            self.records.pop(id(req), None)

    def take(self, req):
        with self.lock:
            record = self.records.pop(id(req), None)
        if record is None or req.err or record["end_ns"] is None:
            return None
        prompt, cached = record["prompt"], record["cached"]
        ms = (record["end_ns"] - record["start_ns"]) / 1e6
        if (not isinstance(prompt, int) or not isinstance(cached, int) or
            not 0 <= cached < prompt or prompt != req.n_prompt or cached != req.start or
            not math.isfinite(ms) or ms <= 0):
            return None
        processed = prompt - cached
        return {"prompt_n": processed, "prompt_ms": ms, "prompt_per_second": processed * 1000 / ms,
                "cache_n": cached, "scope": "server_prefill_wall"}

class PrefillTimingMixin:
    """Observe existing scheduler boundaries without changing its scheduling logic."""
    def __init__(self, *args, **kwargs):
        self.prefill_telemetry = PrefillTelemetry()
        super().__init__(*args, **kwargs)

    def _prefill_start(self, req, slot):
        self.prefill_telemetry.begin(req)
        return super()._prefill_start(req, slot)

    def _prefill_wave(self, wave):
        for req, _ in wave:
            self.prefill_telemetry.begin(req)
        return super()._prefill_wave(wave)

    def _activate(self, req, slot, seed):
        self.prefill_telemetry.finish(req)
        return super()._activate(req, slot, seed)

class TokenHistory:
    """Bounded, process-local mapping from API history to original inference tokens.

    The engine can continue only an exact consumed prefix, including model-generated
    whitespace and tool syntax. API fields are an interchange representation, not a
    lossless serialization of those tokens. Keep the original tokens after a complete
    response, keyed by the full returned message history and rendering configuration.
    This holds token IDs/template text only; it does not save GPU/KV/recurrent state.
    """
    def __init__(self, max_entries=8, max_bytes=32 * 2**20):
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.payload_bytes = 0
        self.lock = threading.Lock()
        self.hits = self.preserved_tokens = self.template_guard_misses = 0

    @staticmethod
    def argument_value(value):
        # JavaScript clients serialize JSON 1.0/-0.0 as 1/0. Preserve numeric
        # meaning in the lookup key without changing the actual inference tokens.
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            return int(value)
        if isinstance(value, list):
            return [TokenHistory.argument_value(x) for x in value]
        if isinstance(value, dict):
            return {k: TokenHistory.argument_value(v) for k, v in value.items()}
        return value

    @staticmethod
    def message_value(message):
        # Match fields used by this Qwen template. Preserve literal text; only
        # normalize interchangeable API forms (null/empty, JSON argument encoding).
        role = message.get("role")
        value = {"role": role, "content": message.get("content")}
        if value["content"] is None or isinstance(value["content"], str) and not value["content"].strip():
            value["content"] = ""
        if role == "assistant":
            reasoning = message.get("reasoning_content")
            value["reasoning_content"] = reasoning if isinstance(reasoning, str) and reasoning.strip() else ""
            value["tool_calls"] = []
            for call in message.get("tool_calls") or []:
                fn = call.get("function", call)
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                value["tool_calls"].append({"id": call.get("id"), "name": fn.get("name"),
                                            "arguments": TokenHistory.argument_value(args)})
        elif role == "tool":
            value["tool_call_id"] = message.get("tool_call_id")
        return value

    @classmethod
    def keys(cls, messages, tools, opts):
        # Keep tool-schema serialization order: it affects the system prompt.
        context = json.dumps([tools, {k: opts[k] for k in sorted(opts)}], ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(context.encode())
        for index, message in enumerate(messages):
            raw = json.dumps(cls.message_value(message), ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode()
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
            yield index + 1, digest.digest()

    def resolve(self, tok, messages, tools, opts, canonical_prompt):
        keys = list(self.keys(messages, tools, opts))
        with self.lock:
            candidates = [(key, self.entries[key]) for _, key in reversed(keys) if key in self.entries]
        for key, entry in candidates:
            prefix = entry["canonical_prefix"]
            if not canonical_prompt.startswith(prefix):
                # Equivalent JSON arguments may arrive with a different key order.
                # Render the matching API prefix again, but still require the full
                # template to extend it. This rejects context-dependent rewriting,
                # e.g. preserve_thinking=false removing old reasoning on a new user turn.
                prefix = tok.apply_chat_template(messages[:entry["messages"]], tools=tools,
                    tokenize=False, add_generation_prompt=False, **opts)
            if not canonical_prompt.startswith(prefix):
                with self.lock:
                    self.template_guard_misses += 1
                continue
            suffix = canonical_prompt[len(prefix):]
            ids = list(entry["ids"])
            ids.extend(tok(entry["separator"] + suffix, add_special_tokens=False).input_ids)
            preserved = len(entry["ids"])
            with self.lock:
                if key in self.entries:
                    self.entries.move_to_end(key)
                self.hits += 1
                self.preserved_tokens += preserved
            return ids, preserved
        return list(tok(canonical_prompt, add_special_tokens=False).input_ids), 0

    def remember(self, tok, messages, tools, opts, prompt_ids, output_ids, eos, assistant):
        history = messages + prepare_messages([assistant])
        prefix = tok.apply_chat_template(history, tools=tools, tokenize=False,
                                        add_generation_prompt=False, **opts)
        if not prefix.endswith("<|im_end|>\n"):
            return False  # A changed template must be requalified before replay.
        key = next(key for n, key in self.keys(history, tools, opts) if n == len(history))
        ids = array("i", prompt_ids)
        ids.extend(output_ids)
        # Native decode includes its final emitted EOS but has not consumed that
        # token yet. Keep it verbatim; append the template's separating newline.
        # A length-limited answer has no EOS: append its closing boundary as input.
        separator = "\n" if output_ids and output_ids[-1] in eos else "<|im_end|>\n"
        size = len(ids) * ids.itemsize + len(prefix.encode())
        if size > self.max_bytes:
            return False
        entry = {"ids": ids, "canonical_prefix": prefix, "separator": separator,
                 "messages": len(history), "bytes": size}
        with self.lock:
            old = self.entries.pop(key, None)
            if old:
                self.payload_bytes -= old["bytes"]
            self.entries[key] = entry
            self.payload_bytes += size
            while self.entries and (len(self.entries) > self.max_entries or self.payload_bytes > self.max_bytes):
                _, old = self.entries.popitem(last=False)
                self.payload_bytes -= old["bytes"]
        return True

    def stats(self):
        with self.lock:
            return {"entries": len(self.entries), "payload_bytes": self.payload_bytes,
                    "max_payload_bytes": self.max_bytes, "hits": self.hits,
                    "preserved_tokens": self.preserved_tokens,
                    "template_guard_misses": self.template_guard_misses}

def make_handler(eng, tok, api_key, parse_tools, config=None):
    config = config or ServerConfig()
    model, context, max_output = config.model, config.context_window, config.max_output
    history = TokenHistory()
    eos = {tok.eos_token_id}
    for name in ["<|im_end|>", "<|endoftext|>"]:
        v = tok.convert_tokens_to_ids(name)
        if v is not None and v >= 0:
            eos.add(v)
    eos.discard(None)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.connection.settimeout(1800)

        def log_message(self, fmt, *values):
            # No prompts, headers, tool arguments, or credentials in access logs.
            pass

        def send_json(self, status, body):
            raw = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def authorized(self):
            if self.path.split("?", 1)[0] == "/health":
                return True
            given = self.headers.get("Authorization", "")
            ok = hmac.compare_digest(given.encode(), ("Bearer " + api_key).encode())
            if not ok:
                self.close_connection = True
                self.send_json(401, {"error": {"message": "Missing or invalid bearer credential", "type": "authentication_error"}})
            return ok

        def do_GET(self):
            if not self.authorized():
                return
            if self.path == "/health":
                self.send_json(200, {"status": "ok", "model": model, "context_window": context,
                    "active": len(eng.active), "queued": len(eng.q), "speculative": eng.spec_on,
                    "draft_vocab": eng.draft_vocab, "pool_tokens": eng.num_blocks * eng.page,
                    "sampling": "temperature_top_k_top_p_min_p", "seeded_sampling": True,
                    "spec_sampling": getattr(eng, "spec_sampling", "probability"), "prefilling": len(eng.prefills),
                    "engine_configuration": getattr(eng, "runtime_info", {}),
                    "embedding": "bf16_gpu", "kv": "trellis4", "host_kv_offload": False,
                    "prefilled_tokens": eng.prefilled_tokens, "decoded_tokens": eng.decoded_tokens,
                    "cache_hits": eng.cache_hits, "cache_tokens": eng.cache_tokens,
                    "token_history": history.stats(),
                    "prefill_timing": "server_prefill_wall" if hasattr(eng, "prefill_telemetry") else None})
            elif self.path.rstrip("/") == "/v1/models":
                self.send_json(200, {"object": "list", "data": [{"id": model, "object": "model", "owned_by": "local",
                    "context_window": context, "max_tokens": max_output, "sampling": "temperature_top_k_top_p_min_p"}]})
            else:
                self.send_json(404, {"error": {"message": "Unknown endpoint"}})

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != "/v1/chat/completions":
                self.send_json(404, {"error": {"message": "Use /v1/chat/completions"}})
                return
            streaming = False
            req = None
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 128 * 2**20:
                    self.close_connection = True
                    self.send_json(413, {"error": {"message": "Invalid or excessive request body"}})
                    return
                body = json.loads(self.rfile.read(n))
                if not isinstance(body, dict):
                    raise ValueError("Request must be a JSON object")
                if body.get("model", model) != model:
                    raise ValueError("Unknown model; use /v1/models for discovery")
                sampling = sampling_options(body)
                limit = int(body.get("max_tokens", body.get("max_completion_tokens", max_output)))
                if not 1 <= limit <= max_output:
                    raise ValueError(f"max_tokens must be between 1 and {max_output}")
                opts = template_options(body)
                tools = body.get("tools") or None
                if body.get("tool_choice") == "none":
                    tools = None
                msgs = prepare_messages(body.get("messages", []))
                prompt = tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, **opts)
                ids, preserved = history.resolve(tok, msgs, tools, opts, prompt)
                if len(ids) + limit > context:
                    raise ValueError(f"Prompt ({len(ids)}) plus requested output ({limit}) exceeds {context} tokens")
                req = eng.submit(list(ids), limit, list(eos), sampling=sampling)
                cid = "chatcmpl-" + uuid.uuid4().hex
                created = int(time.time())
                base = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model}
                stream = bool(body.get("stream", False))
                usage_requested = bool((body.get("stream_options") or {}).get("include_usage"))

                def event(data):
                    self.wfile.write(("data: " + json.dumps(data, ensure_ascii=False) + "\n\n").encode())
                    self.wfile.flush()

                def delta(data):
                    if data:
                        event({**base, "choices": [{"index": 0, "delta": data, "finish_reason": None}]})

                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    streaming = True
                    delta({"role": "assistant"})
                output = []
                sent_reasoning = sent_content = 0
                submitted = time.time()
                last_keepalive = submitted
                while True:
                    try:
                        token = req.tokens.get(timeout=1)
                    except queue.Empty:
                        if time.time() - submitted > 1800:
                            raise TimeoutError("Generation exceeded the 30-minute request limit")
                        if stream and time.time() - last_keepalive >= 5:
                            self.wfile.write(b": keep-alive\n\n")
                            self.wfile.flush()
                            last_keepalive = time.time()
                        continue
                    if token is None:
                        break
                    if token in eos:
                        continue
                    output.append(token)
                    if stream:
                        text = tok.decode(output, skip_special_tokens=False)
                        if text.endswith("\ufffd"):
                            continue
                        reasoning, content = split_reasoning(text, opts["enable_thinking"])
                        if len(reasoning) > sent_reasoning:
                            delta({"reasoning_content": reasoning[sent_reasoning:]})
                            sent_reasoning = len(reasoning)
                        if not tools and len(content) > sent_content:
                            delta({"content": content[sent_content:]})
                            sent_content = len(content)
                if req.err:
                    raise RuntimeError(req.err)
                timer = getattr(eng, "prefill_telemetry", None)
                timing = timer.take(req) if timer is not None else None
                timing_fields = {"timings": timing} if timing is not None else {}
                text = tok.decode(output, skip_special_tokens=False)
                reasoning, content = split_reasoning(text, opts["enable_thinking"], final=True)
                calls = None
                if tools:
                    content, calls = parse_tools(content, tools)
                message = {"role": "assistant", "content": content or None}
                if reasoning:
                    message["reasoning_content"] = reasoning
                if calls:
                    message["tool_calls"] = calls
                # Publish before the terminal response/SSE events so an immediate
                # tool continuation can find the exact completed token history.
                history.remember(tok, msgs, tools, opts, ids, req.out, eos, message)
                finish = "tool_calls" if calls else ("length" if len(req.out) >= limit else "stop")
                usage = {"prompt_tokens": req.n_prompt, "completion_tokens": len(req.out),
                    "total_tokens": req.n_prompt + len(req.out), "prompt_tokens_details": {"cached_tokens": req.start}}
                ttft = req.t_first - req.t_submit
                decode_s = max(req.t_done - req.t_first, 0)
                metrics = {"sampling": "greedy" if sampling[0] == 0 else "stochastic",
                    "sampling_parameters": dict(zip(["temperature", "top_p", "top_k", "min_p", "seed"], sampling)),
                    "context_window": context, "prefix_cached_tokens": req.start,
                    "token_history_preserved_tokens": preserved,
                    "server_ttft_s": ttft, "decode_seconds": decode_s,
                    "decode_tokens_per_second": max(0, len(req.out)-1) / decode_s if decode_s > 0 else None}
                if stream:
                    delta({"reasoning_content": reasoning[sent_reasoning:]} if len(reasoning) > sent_reasoning else {})
                    if len(content) > sent_content:
                        delta({"content": content[sent_content:]})
                    if calls:
                        delta({"tool_calls": [{"index": i, **call} for i, call in enumerate(calls)]})
                    event({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "x_qwentin": metrics, **timing_fields})
                    if usage_requested:
                        event({**base, "choices": [], "usage": usage, "x_qwentin": metrics, **timing_fields})
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    self.send_json(200, {**base, "object": "chat.completion", "choices": [{"index": 0, "message": message,
                        "finish_reason": finish}], "usage": usage, "x_qwentin": metrics, **timing_fields})
                print(json.dumps({"event": "completed", "prompt_tokens": req.n_prompt, "completion_tokens": len(req.out),
                    "cached_tokens": req.start, "finish_reason": finish, **metrics, **timing_fields}), flush=True)
            except (BrokenPipeError, ConnectionResetError):
                if req is not None:
                    if not eng.cancel_queued(req):
                        req.max_new = min(req.max_new, len(req.out) + 1)
            except Exception as exc:
                if req is not None and not req.done.is_set():
                    if not eng.cancel_queued(req):
                        req.max_new = min(req.max_new, len(req.out) + 1)
                if streaming:
                    event({"error": {"message": str(exc), "type": "generation_error"}})
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    self.send_json(400 if isinstance(exc, (ValueError, TypeError, KeyError)) else 500,
                        {"error": {"message": str(exc), "type": "invalid_request_error" if isinstance(exc, ValueError) else "generation_error"}})
            finally:
                timer = getattr(eng, "prefill_telemetry", None)
                if req is not None and timer is not None:
                    timer.discard(req)
    return Handler

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tqf", required=True)
    ap.add_argument("--api-key-file", required=True)
    ap.add_argument("--model-id", default=MODEL)
    ap.add_argument("--context-window", type=int, default=CONTEXT)
    ap.add_argument("--max-output", type=int, default=MAX_OUTPUT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--num-blocks", type=int, default=2200)
    ap.add_argument("--spec-sampling", choices=["probability", "deterministic"], default="probability")
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--spec-k", choices=["auto", "1", "2", "3", "4", "5", "6"], default="auto")
    ap.add_argument("--draft-vocab", choices=["reduced", "full"], default="reduced")
    args = ap.parse_args(argv)
    if not 128 <= args.context_window <= CONTEXT or args.context_window % 128:
        ap.error(f"--context-window must be a multiple of 128 between 128 and {CONTEXT}")
    if not 1 <= args.max_output < args.context_window:
        ap.error("--max-output must be positive and smaller than --context-window")
    if args.num_blocks * 128 < args.context_window:
        ap.error("--num-blocks provides less KV capacity than --context-window")
    return args

def create_engine(args):
    os.environ.update(TQ_CTX=str(args.context_window), TQ_EMBED_HOST="0", TQ_EMBED_FP8="0", TQ_PARK_RING="0")
    sys.path.insert(0, args.source)
    from tools.serve_batched import BatchedEngine, load_lib
    from .tool_parser import parse_tool_calls
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    lib = load_lib(str(Path(args.source) / "build-qwen/libforward_qwen.so"))
    import ctypes
    lib.quinlan_spec_set_mode.argtypes = [ctypes.c_int]
    lib.quinlan_spec_set_mode.restype = None
    lib.quinlan_spec_set_mode(args.spec_sampling == "probability")
    cfg = {"TQ_WIDE_ATTN_MMA": 1, "TQ_EMBED_HOST": 0, "TQ_NO_MTP": 0, "TQ_KV_V4": 1,
           "TQ_KV_T4": 1, "TQ_PDL": 1, "TQ_DSTATE_BF16": 4, "TQ_GDN_PREFILL": 1,
           "TQ_DRAFT_VOCAB": int(args.draft_vocab == "reduced")}
    class TimedEngine(PrefillTimingMixin, BatchedEngine):
        pass
    eng = TimedEngine(lib, args.tqf, 1, args.num_blocks, 128, prefill_waves_per_step=2, mixed_prefill=True,
        prefix_cache=True, park_ram_bytes=0, decode_steps_per_wave=1, spec=not args.no_spec,
        spec_k=args.spec_k, cfg_overrides=cfg)
    eng.spec_sampling = args.spec_sampling
    from tools.serve_batched import qwentin_abi
    actual = qwentin_abi.Config()
    lib.qwn_config_get(ctypes.byref(actual))
    expected = dict(ctx=args.context_window, embed_host=0, embed_fp8=0, park_ring=0, kv_t4=1, kv_v4=1, dstate=4)
    for field, value in expected.items():
        if getattr(actual, field) != value:
            eng.shutdown()
            raise RuntimeError(f"Effective {field} differs from qualified configuration")
    lib.qwn_device_bytes.restype = ctypes.c_size_t
    lib.qwn_max_seq.restype = ctypes.c_int
    if lib.qwn_max_seq() != args.context_window or eng.max_blocks_per_seq * eng.page < args.context_window:
        eng.shutdown()
        raise RuntimeError("Engine context or KV pool capacity differs from the requested configuration")
    eng.runtime_info = {**expected, "weight_device_bytes": lib.qwn_device_bytes(),
        "max_blocks_per_sequence": eng.max_blocks_per_seq, "pool_blocks": eng.num_blocks,
        "host_park_budget": eng.park_ram_bytes}
    return eng, tok, parse_tool_calls

def main():
    args = parse_args()
    key = Path(args.api_key_file).read_text().strip()
    if len(key) < 24:
        raise ValueError("API key must contain at least 24 characters")
    eng, tok, parse_tool_calls = create_engine(args)
    config = ServerConfig(args.model_id, args.context_window, args.max_output)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(eng, tok, key, parse_tool_calls, config))
    print(json.dumps({"event": "ready", "model": config.model, "context_window": config.context_window, "port": args.port,
        "all_model_weights_in_vram": True, "host_kv_offload": False, "sampling": "temperature_top_k_top_p_min_p"}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        eng.shutdown()

if __name__ == "__main__":
    main()
