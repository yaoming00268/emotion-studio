"""Cache public, pinned source files; verify Hub Git/LFS identities."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

from ..common import atomic_json, now

USER_AGENT = "EmotionStudio-Corpus/1.0 (local research dataset; cached downloads)"


def read_url(url, *, headers=None, maximum=64 * 1024**2):
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})}), timeout=60) as response:
                data = response.read(maximum + 1)
                if len(data) > maximum:
                    raise ValueError("Source exceeds download size limit")
                return data, dict(response.headers), response.status
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def fetch(url, destination, *, expected_sha256=None, git_oid=None, size=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def valid(data):
        if size is not None and len(data) != size:
            return False
        if expected_sha256 and hashlib.sha256(data).hexdigest() != expected_sha256:
            return False
        if git_oid and hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest() != git_oid:
            return False
        return True

    receipt = destination.with_name(destination.name + ".receipt.json")
    if destination.is_file() and receipt.is_file():
        stored = json.loads(receipt.read_text(encoding="utf-8"))
        data = destination.read_bytes()
        if stored.get("url") == url and stored.get("sha256") == hashlib.sha256(data).hexdigest() and valid(data):
            return stored
    data, _, _ = read_url(url)
    if not valid(data):
        raise ValueError("Source checksum mismatch: " + url)
    temporary = destination.with_name(destination.name + ".pending")
    temporary.write_bytes(data)
    temporary.replace(destination)
    result = {"url": url, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "fetched_at": now()}
    atomic_json(receipt, result)
    return result


def prepare_teacher(destination):
    return prepare_model_snapshot(destination,
        repo="tabularisai/multilingual-emotion-classification",
        revision="b70346f3d6015955e4ae7fdd345358f0811190a7", license_name="MIT")


def prepare_model_snapshot(destination, *, repo, revision, license_name):
    """Download only inference assets, checking both Git and LFS identities."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    listing = json.loads(read_url(f"https://huggingface.co/api/models/{repo}/tree/{revision}")[0])
    base = f"https://huggingface.co/{repo}/resolve/{revision}/"
    receipts = []
    for entry in listing:
        name = entry["path"]
        if name not in {"config.json", "tokenizer.json", "tokenizer_config.json", "README.md", "model.safetensors",
                        "special_tokens_map.json", "ekman_mapping.json", "thresholds.json", "LICENSE", "NOTICE", "SHA256SUMS",
                        "added_tokens.json", "spm.model"}:
            continue
        lfs = entry.get("lfs", {})
        digest = lfs.get("oid")
        if name != "model.safetensors":
            receipts.append(fetch(base + name, destination / name, expected_sha256=digest,
                                  git_oid=None if lfs else entry["oid"], size=entry["size"]))
            continue
        target = destination / name
        if target.is_file() and target.stat().st_size == entry["size"]:
            with target.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() == digest:
                    continue
        parts = destination / ".download"
        parts.mkdir(exist_ok=True)
        block, total = 8 * 1024**2, entry["size"]
        count = (total + block - 1) // block

        def download(index):
            start, end = index * block, min(total, (index + 1) * block) - 1
            path = parts / f"{index:04}.part"
            if path.is_file() and path.stat().st_size == end - start + 1:
                return
            data, headers, status = read_url(base + name + f"?emotionstudio_range={start}",
                headers={"Range": f"bytes={start}-{end}"}, maximum=block)
            content_range = next((v for k, v in headers.items() if k.lower() == "content-range"), None)
            if status != 206 or content_range != f"bytes {start}-{end}/{total}" or len(data) != end - start + 1:
                raise ValueError("Teacher download did not honor byte range")
            path.write_bytes(data)

        with ThreadPoolExecutor(max_workers=6) as pool:
            for done, job in enumerate(as_completed([pool.submit(download, i) for i in range(count)]), 1):
                job.result()
                if done % 8 == 0 or done == count:
                    print(f"Teacher weights: {done}/{count} chunks", flush=True)
        temporary = target.with_name(target.name + ".pending")
        hasher = hashlib.sha256()
        with temporary.open("wb") as stream:
            for index in range(count):
                data = (parts / f"{index:04}.part").read_bytes()
                stream.write(data)
                hasher.update(data)
        if hasher.hexdigest() != digest:
            raise ValueError("Teacher SHA256 mismatch")
        temporary.replace(target)
        # Only the exactly enumerated generated download parts are removed.
        for index in range(count):
            (parts / f"{index:04}.part").unlink()
        parts.rmdir()
    atomic_json(destination / "source.json", {"repo": repo, "revision": revision, "license": license_name,
                "downloaded_at": now(), "files": listing})
    return destination
