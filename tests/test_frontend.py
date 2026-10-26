"""CPU integration checks for the Pi adapter, with the upstream tool parser."""
import ast
import importlib.util
import json
from pathlib import Path
import queue
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import urllib.error
import urllib.request
import uuid

from quinlan import server as adapter
from quinlan.tool_parser import parse_tool_calls

class FakeTokenizer:
    eos_token_id = 0
    def __init__(self):
        self.options = None
    def convert_tokens_to_ids(self, name):
        return 0
    def apply_chat_template(self, messages, **kwargs):
        self.options = (messages, kwargs)
        return "x" * (adapter.CONTEXT if messages[0].get("content") == "overflow" else 10)
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[1] * len(text))
    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)

class FakeEngine:
    def __init__(self):
        self.active = {}; self.q = []; self.spec_on = True; self.draft_vocab = 65536
        self.prefills = []
        self.num_blocks = 2600; self.page = 128; self.prefilled_tokens = 0; self.decoded_tokens = 0
        self.cache_hits = 0; self.cache_tokens = 0; self.submits = 0
        self.prefill_clock_ns = 0
        self.prefill_telemetry = adapter.PrefillTelemetry(clock_ns=lambda: self.prefill_clock_ns)
        self.text = 'Check the evidence.\n</think>\n<tool_call>{"name":"lookup","arguments":{"config":{"ids":[2,7]}}}</tool_call>'
    def submit(self, ids, limit, eos, sampling=None):
        self.submits += 1
        self.sampling = sampling
        now = time.time()
        req = SimpleNamespace(tokens=queue.Queue(), out=[ord(c) for c in self.text], n_prompt=len(ids), start=3,
            t_first=now+0.01, t_submit=now, t_done=now+1, err=None, max_new=limit, done=threading.Event())
        self.prefill_clock_ns += 1000000
        self.prefill_telemetry.begin(req)
        self.prefill_clock_ns += 25000000
        self.prefill_telemetry.finish(req)
        for t in req.out:
            req.tokens.put(t)
        req.tokens.put(None); req.done.set()
        return req
    def cancel_queued(self, req):
        return False

class FrontendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        cls.eng = FakeEngine(); cls.tok = FakeTokenizer()
        parser = parse_tool_calls
        cls.server = adapter.ThreadingHTTPServer(("127.0.0.1", 0), adapter.make_handler(cls.eng, cls.tok, "test-key", parser))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True); cls.thread.start()
        cls.url = "http://127.0.0.1:" + str(cls.server.server_port)
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()
    def post(self, body):
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Authorization":"Bearer test-key", "Content-Type":"application/json"})
        return urllib.request.urlopen(req)
    def test_authentication(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(self.url + "/v1/models")
        self.assertEqual(caught.exception.code, 401)
        caught.exception.close()
    def test_unknown_model_is_rejected_before_submission(self):
        before = self.eng.submits
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post({"model":"missing", "messages":[{"role":"user", "content":"hello"}]})
        self.assertEqual(caught.exception.code, 400)
        caught.exception.close()
        self.assertEqual(self.eng.submits, before)
    def test_configurable_model_and_context_discovery(self):
        config = adapter.ServerConfig("custom-model", 128, 32)
        server = adapter.ThreadingHTTPServer(("127.0.0.1", 0),
            adapter.make_handler(self.eng, self.tok, "test-key", parse_tool_calls, config))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = "http://127.0.0.1:" + str(server.server_port)
            request = urllib.request.Request(base + "/v1/models", headers={"Authorization":"Bearer test-key"})
            with urllib.request.urlopen(request) as response:
                model = json.load(response)["data"][0]
            self.assertEqual((model["id"], model["context_window"], model["max_tokens"]), ("custom-model",128,32))
            request = urllib.request.Request(base + "/v1/chat/completions",
                data=json.dumps({"model":"custom-model", "messages":[{"role":"user", "content":"hello"}]}).encode(),
                headers={"Authorization":"Bearer test-key", "Content-Type":"application/json"})
            with urllib.request.urlopen(request) as response:
                self.assertEqual(json.load(response)["model"], "custom-model")
        finally:
            server.shutdown(); server.server_close(); thread.join()
    def test_fragmented_reasoning_delimiter(self):
        raw = "Detailed reasoning.\n</think>\nFinal answer."
        last_r = last_c = ""
        for n in range(1, len(raw)+1):
            r, c = adapter.split_reasoning(raw[:n], True)
            self.assertTrue(r.startswith(last_r)); self.assertTrue(c.startswith(last_c))
            last_r, last_c = r, c
        self.assertEqual((last_r, last_c), ("Detailed reasoning.\n", "Final answer."))
    def test_streaming_reasoning_nested_tool_and_usage(self):
        body = {"messages":[{"role":"user", "content":"lookup"}], "stream":True, "stream_options":{"include_usage":True},
            "tools":[{"type":"function", "function":{"name":"lookup", "parameters":{"type":"object", "properties":{"config":{"type":"object"}}}}}]}
        events = [json.loads(line[6:]) for line in self.post(body).read().decode().splitlines() if line.startswith("data: {")]
        reasoning = "".join(e.get("choices", [{}])[0].get("delta", {}).get("reasoning_content", "") for e in events if e.get("choices"))
        self.assertEqual(reasoning, "Check the evidence.\n")
        calls = [e["choices"][0]["delta"]["tool_calls"][0] for e in events if e.get("choices") and "tool_calls" in e["choices"][0].get("delta", {})]
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"config":{"ids":[2,7]}})
        self.assertEqual(events[-1]["usage"]["prompt_tokens_details"]["cached_tokens"], 3)
        self.assertTrue(self.tok.options[1]["enable_thinking"])
    def test_tool_history_arguments_and_effort(self):
        body = {"messages":[{"role":"assistant", "content":None, "tool_calls":[{"function":{"name":"lookup", "arguments":'{"config":{"ids":[2,7]}}'}}]},
            {"role":"tool", "content":"result"}], "reasoning_effort":"medium"}
        self.post(body).read()
        messages, opts = self.tok.options
        self.assertIsInstance(messages[0]["tool_calls"][0]["function"]["arguments"], dict)
        self.assertEqual(opts["reasoning_effort"], "medium")
    def test_streamed_tool_preamble_matches_preserved_history(self):
        old = self.eng.text
        self.eng.text = 'Check this.\n</think>\nI will inspect the file now.\n<tool_call>{"name":"lookup","arguments":{"config":{"ids":[2,7]}}}</tool_call>'
        try:
            body = {"messages":[{"role":"user","content":"inspect"}],"stream":True,
                "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"config":{"type":"object"}}}}}]}
            with mock.patch.object(adapter.TokenHistory, "remember", autospec=True, return_value=True) as remember:
                events = [json.loads(line[6:]) for line in self.post(body).read().decode().splitlines() if line.startswith("data: {")]
                preserved = remember.call_args.args[-1]
            deltas = [c.get("delta", {}) for e in events for c in e.get("choices", [])]
            content = "".join(d.get("content", "") for d in deltas)
            reasoning = "".join(d.get("reasoning_content", "") for d in deltas)
            calls = [{k:v for k,v in call.items() if k != "index"} for d in deltas for call in d.get("tool_calls", [])]
            self.assertEqual(content, "I will inspect the file now.")
            self.assertEqual(preserved, {"role":"assistant", "content":content,"reasoning_content":reasoning,"tool_calls":calls})
        finally:
            self.eng.text = old
    def test_sampling_propagated_to_engine(self):
        self.post({"messages":[{"role":"user", "content":"hello"}], "temperature":0.7,
            "top_p":0.8, "top_k":40, "min_p":0.1, "seed":-7}).read()
        self.assertEqual(self.eng.sampling, (0.7, 0.8, 40, 0.1, 2**64-7))
    def test_invalid_sampling_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post({"messages":[{"role":"user", "content":"hello"}], "temperature":-1})
        self.assertEqual(caught.exception.code, 400)
    def test_context_admission(self):
        old = self.eng.submits
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post({"messages":[{"role":"user", "content":"overflow"}], "max_tokens":1})
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(self.eng.submits, old)
    def test_streaming_prefill_compatibility_fields(self):
        events = [json.loads(line[6:]) for line in self.post({"messages":[{"role":"user","content":"timing"}],
            "stream":True,"stream_options":{"include_usage":True}}).read().decode().splitlines() if line.startswith("data: {")]
        final = next(e for e in events if e.get("choices") and e["choices"][0]["finish_reason"])
        timing = {"prompt_n":7,"prompt_ms":25.0,"prompt_per_second":280.0,"cache_n":3,"scope":"server_prefill_wall"}
        self.assertEqual(final["timings"], timing)
        self.assertEqual(events[-1]["timings"], timing)
        self.assertEqual(events[-1]["usage"]["prompt_tokens_details"]["cached_tokens"], timing["cache_n"])
    def test_nonstream_prefill_compatibility_fields(self):
        result = json.load(self.post({"messages":[{"role":"user","content":"timing"}]}))
        self.assertEqual(result["timings"]["prompt_n"],result["usage"]["prompt_tokens"]-3)
        self.assertEqual(result["timings"]["prompt_ms"],25)
    def test_streaming_prefill_without_usage_opt_in(self):
        events = [json.loads(line[6:]) for line in self.post({"messages":[{"role":"user","content":"timing"}],
            "stream":True}).read().decode().splitlines() if line.startswith("data: {")]
        self.assertEqual(events[-1]["timings"]["prompt_per_second"],280)
        self.assertNotIn("usage",events[-1])
    def test_missing_prefill_measurement_never_uses_ttft(self):
        with mock.patch.object(adapter.PrefillTelemetry, "take", return_value=None):
            result = json.load(self.post({"messages":[{"role":"user","content":"timing"}]}))
        self.assertGreater(result["x_qwentin"]["server_ttft_s"],0)
        self.assertNotIn("timings",result)

if __name__ == "__main__":
    unittest.main()
