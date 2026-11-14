"""Explicit download step. Inference and training are otherwise local-files-only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen

from .common import BASE_MODEL_ID, BASE_REVISION, PROJECT_ROOT, atomic_json, now


FILES = {
    "config.json": (624, "a521dc2845bdddbe822864290c6b928396fc5ee8"),
    "tokenizer.json": (268943, "dd9401af0a4713df0e75066ae78d5bf6fa6c7116"),
    "tokenizer_config.json": (49, "2ba5de7675473164e07f3b3531748c9a6f113a2c"),
    "vocab.txt": (109540, "ca4f9781030019ab9b253c6dcb8c7878b6dc87a5"),
}
WEIGHT_SIZE = 411553788
WEIGHT_SHA256 = "3404a1ffd8da507042e8161013ba2a4fc49858b4e3f8fbf5ce5724f94883aec3"


def prepare(status_path):
    status = Path(status_path)
    destination = PROJECT_ROOT / "models/bert-base-chinese"
    atomic_json(status, {"state": "downloading", "message": "下载固定版本中文 BERT 权重和分词器，约 400MB。"})
    try:
        destination.mkdir(parents=True, exist_ok=True)
        base_url = f"https://huggingface.co/{BASE_MODEL_ID}/resolve/{BASE_REVISION}/"
        for name, (size, blob) in FILES.items():
            with urlopen(base_url + name + "?download=true", timeout=60) as response:
                data = response.read(size + 1)
            digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if len(data) != size or digest != blob:
                raise ValueError("基础模型文件校验失败：" + name)
            (destination / name).write_bytes(data)
        target = destination / "model.safetensors"
        valid = target.is_file() and target.stat().st_size == WEIGHT_SIZE
        if valid:
            with target.open("rb") as stream:
                valid = hashlib.file_digest(stream, "sha256").hexdigest() == WEIGHT_SHA256
        if not valid:
            parts = destination / ".download"
            parts.mkdir(exist_ok=True)
            block = 8 * 1024**2
            count = (WEIGHT_SIZE + block - 1) // block

            def fetch(index):
                start, end = index * block, min((index + 1) * block, WEIGHT_SIZE) - 1
                part = parts / f"{index:03}.part"
                if part.is_file() and part.stat().st_size == end - start + 1:
                    return
                for attempt in range(4):
                    try:
                        req = Request(base_url + f"model.safetensors?download=true&range={start}",
                                      headers={"Range": f"bytes={start}-{end}"})
                        with urlopen(req, timeout=60) as response:
                            if response.status != 206 or response.getheader("Content-Range") != f"bytes {start}-{end}/{WEIGHT_SIZE}":
                                raise ValueError("下载服务未返回完整的请求分段。")
                            data = response.read(end - start + 2)
                        if len(data) != end - start + 1:
                            raise ValueError("模型分段下载不完整。")
                        part.write_bytes(data)
                        return
                    except Exception:
                        if attempt == 3:
                            raise
                        time.sleep(attempt + 1)
            with ThreadPoolExecutor(max_workers=6) as pool:
                for done, result in enumerate(as_completed([pool.submit(fetch, index) for index in range(count)]), 1):
                    result.result()
                    atomic_json(status, {"state": "downloading", "done": done, "total": count,
                                         "message": f"下载 BERT 权重：{done}/{count} 分段。"})
            temporary = destination / "model.safetensors.pending"
            digest = hashlib.sha256()
            with temporary.open("wb") as stream:
                for index in range(count):
                    data = (parts / f"{index:03}.part").read_bytes()
                    stream.write(data)
                    digest.update(data)
            if digest.hexdigest() != WEIGHT_SHA256:
                raise ValueError("BERT 权重 SHA256 校验失败，请删除 .download 分段后重试。")
            temporary.replace(target)
            for index in range(count):
                (parts / f"{index:03}.part").unlink()
            parts.rmdir()
        atomic_json(destination / "source.json", {"repo": BASE_MODEL_ID, "revision": BASE_REVISION, "downloaded_at": now()})
        atomic_json(status, {"state": "completed", "path": str(destination), "message": "基础模型已准备。"})
    except BaseException as error:
        atomic_json(status, {"state": "failed", "error": f"{type(error).__name__}: {error}"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", required=True)
    prepare(parser.parse_args().status)
