"""Resume official Windows CUDA wheel with verified HTTP ranges and SHA256."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time
from urllib.request import Request, build_opener, ProxyHandler

URL = "https://download.pytorch.org/whl/cu128/torch-2.8.0%2Bcu128-cp312-cp312-win_amd64.whl"
EXPECTED = "0ad925202387f4e7314302a1b4f8860fa824357f9b1466d7992bf276370ebcff"
TOTAL = 3461384651
BLOCK = 16 * 1024**2


def main():
    root = Path(__file__).resolve().parents[1] / "runtime/wheels"
    root.mkdir(parents=True, exist_ok=True)
    target = root / "torch-2.8.0+cu128-cp312-cp312-win_amd64.whl"
    parts = root / "torch-parts"
    parts.mkdir(exist_ok=True)
    count = (TOTAL + BLOCK - 1) // BLOCK

    def download(index):
        start, end = index * BLOCK, min(TOTAL, (index + 1) * BLOCK) - 1
        path = parts / f"{index:04}.part"
        if path.exists() and path.stat().st_size == end - start + 1:
            return index
        for attempt in range(5):
            try:
                # Query prevents intermediate caches from confusing byte ranges.
                request = Request(URL + f"?part={index}", headers={"Range": f"bytes={start}-{end}"})
                with build_opener(ProxyHandler({})).open(request, timeout=50) as response:
                    if response.status != 206 or response.getheader("Content-Range") != f"bytes {start}-{end}/{TOTAL}":
                        raise ValueError("Server did not honor the requested range")
                    data = response.read(end - start + 2)
                    if len(data) != end - start + 1:
                        raise ValueError("Incomplete range")
                path.write_bytes(data)
                return index
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(1 + attempt)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for done, result in enumerate(as_completed([pool.submit(download, index) for index in range(count)]), 1):
            result.result()
            if done % 8 == 0 or done == count:
                print(f"CUDA wheel: {done}/{count} blocks ({min(100, done * 100 // count)}%)", flush=True)
    hasher = hashlib.sha256()
    with target.open("wb") as stream:
        for index in range(count):
            with (parts / f"{index:04}.part").open("rb") as source:
                while chunk := source.read(1024**2):
                    stream.write(chunk)
                    hasher.update(chunk)
    if hasher.hexdigest() != EXPECTED:
        raise ValueError("Official CUDA wheel SHA256 mismatch")
    print(f"SHA256 verified: {target}", flush=True)


if __name__ == "__main__":
    main()
