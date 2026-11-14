"""Lightweight HTTP client; usable without torch, transformers or training code."""
from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener


def request(service, route, payload=None, timeout=180):
    address = service.get("url", "")
    parsed = urlparse(address)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or parsed.path not in ("", "/"):
        raise ValueError("BERT 服务地址必须为本机 HTTP 地址。")
    if not service.get("token"):
        raise ValueError("缺少 BERT 本地服务令牌，请重新启动分析服务。")
    headers = {"X-Emotion-Token": service["token"], "Content-Type": "application/json"}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    req = Request(address.rstrip("/") + route, data=data, headers=headers)
    try:
        with build_opener(ProxyHandler({})).open(req, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        detail = json.loads(error.read().decode("utf-8"))
        raise RuntimeError(detail.get("error", str(error))) from error
