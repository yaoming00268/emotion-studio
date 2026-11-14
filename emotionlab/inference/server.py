"""Authenticated, loopback-only JSON API in the analysis process."""
from __future__ import annotations

import argparse
from functools import partial
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets

from ..common import atomic_json, now, read_json
from .engine import BertAnalyzer


class Handler(BaseHTTPRequestHandler):
    server_version = "EmotionBERT/1"

    def log_message(self, format, *args):
        pass

    def respond(self, status, body):
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        return hmac.compare_digest(self.headers.get("X-Emotion-Token", ""), self.server.token)

    def do_GET(self):
        if not self.authorized():
            return self.respond(401, {"error": "需要本地服务令牌。"})
        if self.path == "/health":
            return self.respond(200, {"ok": True, **self.server.analyzer.status()})
        self.respond(404, {"error": "接口不存在。"})

    def do_POST(self):
        if not self.authorized():
            return self.respond(401, {"error": "需要本地服务令牌。"})
        if self.path != "/predict":
            return self.respond(404, {"error": "接口不存在。"})
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4 * 1024 * 1024:
                return self.respond(413, {"error": "请求必须小于 4MB。"})
            if self.headers.get_content_type() != "application/json":
                return self.respond(415, {"error": "请使用 application/json。"})
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("请求必须是 JSON 对象。")
            results = self.server.analyzer.predict(payload.get("texts"), payload.get("batch_size", 8))
            self.respond(200, {"predictions": results, "model": self.server.analyzer.status()})
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.respond(400, {"error": str(error)})
        except Exception as error:
            self.respond(503, {"error": f"{type(error).__name__}: {error}"})


def make_server(checkpoint, port=18765, device="cuda", token=None, analyzer=None):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.analyzer = analyzer if analyzer is not None else BertAnalyzer(checkpoint, device)
    server.token = token or secrets.token_urlsafe(32)
    return server


def main():
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--service-file", required=True)
    args = parser.parse_args()
    service_file = Path(args.service_file).resolve()
    with make_server(args.checkpoint, args.port, args.device) as server:
        info = {"url": f"http://127.0.0.1:{server.server_address[1]}", "token": server.token,
                "pid": os.getpid(), "checkpoint": str(Path(args.checkpoint).resolve()), "device": args.device,
                "instance_id": os.environ.get("EMOTIONLAB_INSTANCE_ID"),
                "started_at": now()}
        atomic_json(service_file, info)
        try:
            server.serve_forever(poll_interval=.25)
        except KeyboardInterrupt:
            pass
        finally:
            if (read_json(service_file) or {}).get("pid") == os.getpid():
                service_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
