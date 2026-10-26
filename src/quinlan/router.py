"""Small authenticated model-alias router; completion response bytes pass unchanged."""

import argparse
from dataclasses import dataclass
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Route:
    base_url: str
    model: str
    api_key: str = ""
    context_window: int | None = None
    max_output_tokens: int | None = None
    input_modalities: tuple = ("text",)

    def __post_init__(self):
        url = urlsplit(self.base_url)
        if (url.scheme not in {"http", "https"} or not url.hostname or
                url.username or url.password or url.query or url.fragment or not self.model):
            raise ValueError("Route needs an HTTP(S) base URL without credentials and a model ID")
        if not self.input_modalities or any(x not in ('text', 'image') for x in self.input_modalities):
            raise ValueError('Supported input modalities are text and image')


def make_handler(routes, api_key, max_body_bytes=32 * 2**20, timeout=1800):
    if not routes or len(api_key) < 24:
        raise ValueError("Configure routes and a client API key of at least 24 characters")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass  # No request bodies, client addresses, headers, or credentials in logs.

        def json(self, status, value):
            raw = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(raw)

        def error(self, status, message):
            self.json(status, {"error": {"message": message, "type": "router_error"}})

        def authorized(self):
            supplied = self.headers.get("Authorization", "").encode()
            if hmac.compare_digest(supplied, ("Bearer " + api_key).encode()):
                return True
            self.error(401, "Missing or invalid router bearer credential")
            return False

        def do_GET(self):
            if not self.authorized():
                return
            if self.path == "/health":
                self.json(200, {"status": "ok", "service": "quinlan-router"})
            elif self.path.rstrip("/") == "/v1/models":
                data = []
                for alias, route in routes.items():
                    item = {"id": alias, "object": "model", "owned_by": "local"}
                    item['input_modalities'] = list(route.input_modalities)
                    if route.context_window is not None:
                        item["context_window"] = route.context_window
                    if route.max_output_tokens is not None:
                        item["max_output_tokens"] = route.max_output_tokens
                    data.append(item)
                self.json(200, {"object": "list", "data": data})
            else:
                self.error(404, "Unknown endpoint")

        def do_POST(self):
            if not self.authorized():
                return
            if self.path != "/v1/chat/completions":
                self.error(404, "Use /v1/chat/completions")
                return
            connection = None
            started = False
            try:
                if self.headers.get("Transfer-Encoding"):
                    self.error(400, "Send a Content-Length request body")
                    return
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= max_body_bytes:
                    self.error(413, "Invalid or excessive request body")
                    return
                self.connection.settimeout(timeout)
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Request must be a JSON object")
                alias = body.get("model")
                if not isinstance(alias, str) or alias not in routes:
                    self.error(404, "Unknown model; use /v1/models for discovery")
                    return
                route = routes[alias]
                # Keep messages, reasoning, tool calls, schemas, and all extension
                # fields intact. Model alias translation is the sole body rewrite.
                body["model"] = route.model
                payload = json.dumps(body, ensure_ascii=False).encode()
                url = urlsplit(route.base_url)
                cls = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
                connection = cls(url.hostname, url.port, timeout=timeout)
                headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
                if route.api_key:
                    headers["Authorization"] = "Bearer " + route.api_key
                path = url.path.rstrip("/") + "/chat/completions"
                connection.request("POST", path, body=payload, headers=headers)
                response = connection.getresponse()
                # Never forward redirects, cookies, authentication headers, or
                # backend error bodies (which may contain private deployment data).
                if response.status != 200:
                    status = response.status if response.status in {400, 413, 429} else 502
                    self.error(status, f"Backend rejected the request (HTTP {response.status})")
                    return
                kind = response.getheader("Content-Type", "application/json").split(";", 1)[0].strip().lower()
                if kind not in {"application/json", "text/event-stream"}:
                    self.error(502, "Backend returned an unsupported content type")
                    return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                started = True
                while chunk := response.read1(64 * 1024):
                    # Preserve final SSE timings, optional usage events, cache
                    # counters, reasoning and tool deltas byte for byte.
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                self.close_connection = True
            except (ValueError, TypeError):
                if not started:
                    self.error(400, "Invalid JSON request")
            except (OSError, http.client.HTTPException):
                if not started:
                    self.error(502, "Backend connection failed")
            finally:
                if connection is not None:
                    connection.close()

    return Handler


def load_config(path):
    config = json.loads(Path(path).read_text())
    key = os.environ[config["api_key_env"]]
    routes = {}
    for alias, value in config["routes"].items():
        item = dict(value)
        env_name = item.pop("api_key_env", None)
        if "api_key" in item:
            raise ValueError("Use api_key_env instead of storing credentials in router JSON")
        item["api_key"] = os.environ[env_name] if env_name else ""
        routes[alias] = Route(**item)
    return config, routes, key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config, routes, key = load_config(args.config)
    handler = make_handler(routes, key, config.get("max_body_bytes", 32 * 2**20), config.get("timeout", 1800))
    server = ThreadingHTTPServer((config.get("host", "127.0.0.1"), config.get("port", 8787)), handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
