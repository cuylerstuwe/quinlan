"""Exercise the public boundary using a local synthetic backend."""
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

from quinlan import router


class RouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.requests = []
        cls.payload = b'data: {"model":"engine-id","choices":[],"timings":{"prompt_n":7,"prompt_ms":25,"prompt_per_second":280,"cache_n":3,"scope":"server_prefill_wall"},"usage":{"prompt_tokens_details":{"cached_tokens":3}},"unknown_extension":true}\n\ndata: [DONE]\n\n'
        outer = cls

        class Backend(router.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, dict(self.headers), body))
                if body.get("simulate_error"):
                    self.send_response(500)
                    payload = b'private backend diagnostics'
                else:
                    self.send_response(200)
                    payload = outer.payload if body.get("stream") else b'{"model":"engine-id","timings":{"prompt_n":7},"extra":true}'
                self.send_header("Content-Type", "text/event-stream" if body.get("stream") else "application/json")
                self.send_header("Set-Cookie", "private=value")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                # Split an event in the middle to exercise the byte relay.
                self.wfile.write(payload[:13]); self.wfile.flush()
                self.wfile.write(payload[13:])

        cls.backend = router.ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        cls.backend_thread = threading.Thread(target=cls.backend.serve_forever, daemon=True)
        cls.backend_thread.start()
        route = router.Route(f"http://127.0.0.1:{cls.backend.server_port}/v1", "engine-id", "backend-fixture-key")
        cls.server = router.ThreadingHTTPServer(("127.0.0.1", 0), router.make_handler(
            {"public-model": route}, "router-fixture-key-at-least-24", max_body_bytes=4096))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        for server, thread in [(cls.server, cls.thread), (cls.backend, cls.backend_thread)]:
            server.shutdown(); server.server_close(); thread.join()

    def post(self, body):
        request = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(),
            {"Authorization": "Bearer router-fixture-key-at-least-24", "Content-Type": "application/json"})
        return urllib.request.urlopen(request)

    def test_relay_preserves_sse_extensions_and_history(self):
        body = {"model": "public-model", "stream": True, "messages": [
            {"role": "assistant", "content": "  verbatim preamble\n", "reasoning_content": "reasoning\n",
             "tool_calls": [{"id": "call_fixture", "type": "function", "function": {
                 "name": "lookup", "arguments": '{ "x": 1.0 }'}}]},
            {"role": "tool", "tool_call_id": "call_fixture", "content": "result\n"}],
            "chat_template_kwargs": {"preserve_thinking": True}, "seed": 123}
        with self.post(body) as response:
            self.assertEqual(response.read(), self.payload)
            self.assertIsNone(response.headers.get("Set-Cookie"))
        path, headers, forwarded = self.requests[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(headers["Authorization"], "Bearer backend-fixture-key")
        self.assertEqual(forwarded, {**body, "model": "engine-id"})

    def test_nonstream_json_is_passed_unchanged(self):
        with self.post({"model": "public-model", "messages": []}) as response:
            self.assertEqual(response.headers["Content-Type"], "application/json")
            self.assertEqual(response.read(), b'{"model":"engine-id","timings":{"prompt_n":7},"extra":true}')

    def test_image_content_is_forwarded_without_reformatting(self):
        content = [{'type':'image_url','image_url':{'url':'data:image/png;base64,c3ludGhldGlj'}},
                   {'type':'text','text':'Describe the image.'}]
        body = {'model':'public-model','messages':[{'role':'user','content':content}]}
        with self.post(body) as response:
            response.read()
        self.assertEqual(self.requests[-1][2],{**body,'model':'engine-id'})

    def test_auth_and_model_discovery(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(self.url + "/v1/models")
        self.assertEqual(error.exception.code, 401)
        error.exception.close()
        request = urllib.request.Request(self.url + "/v1/models",
            headers={"Authorization": "Bearer router-fixture-key-at-least-24"})
        with urllib.request.urlopen(request) as response:
            value = json.load(response)
        self.assertEqual(value["data"][0]["id"], "public-model")
        self.assertNotIn("base_url", json.dumps(value))

    def test_unknown_model_and_oversized_request_do_not_reach_backend(self):
        for body, status in [({"model": "missing"}, 404),
                             ({"model": "public-model", "padding": "x" * 5000}, 413)]:
            before = len(self.requests)
            with self.assertRaises(urllib.error.HTTPError) as error:
                self.post(body)
            self.assertEqual(error.exception.code, status)
            error.exception.close()
            self.assertEqual(len(self.requests), before)

    def test_backend_errors_are_sanitized(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.post({"model": "public-model", "simulate_error": True})
        self.assertEqual(error.exception.code, 502)
        self.assertNotIn(b"private backend diagnostics", error.exception.read())
        error.exception.close()

    def test_config_resolves_secrets_only_from_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.json"
            config = {"api_key_env": "ROUTER_TEST_KEY", "routes": {"model": {
                "base_url": "http://127.0.0.1:8000/v1", "model": "engine",
                "api_key_env": "BACKEND_TEST_KEY"}}}
            path.write_text(json.dumps(config))
            with patch.dict(os.environ, {"ROUTER_TEST_KEY": "client", "BACKEND_TEST_KEY": "backend"}):
                _, routes, key = router.load_config(path)
                self.assertEqual(key, "client")
                self.assertEqual(routes["model"].api_key, "backend")
                config["routes"]["model"]["api_key"] = "literal-forbidden"
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    router.load_config(path)

    def test_unsafe_urls_are_rejected(self):
        for url in ["file:///tmp/backend", "http://user:pass@localhost/v1", "http://localhost/v1?key=value"]:
            with self.assertRaises(ValueError):
                router.Route(url, "model")
