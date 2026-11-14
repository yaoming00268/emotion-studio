"""Archive scores whose target changed before resuming a rebuilt corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotionlab.common import atomic_json, now, read_json
from emotionlab.corpus.acg_publication import read_rows


def reconcile(root, candidate_file):
    for name in ("annotation-status.json", "verification-status.json"):
        if read_json(root / name, {}).get("state") in {"annotating", "verifying"}:
            raise ValueError("Wait for both scoring processes to finish before reconciling caches")
    targets = {r["id"]: hashlib.sha256(r["text"].encode()).hexdigest() for r in read_rows(candidate_file)}
    digest = hashlib.sha256(candidate_file.read_bytes()).hexdigest()
    report = {"candidate_sha256": digest, "created_at": now(), "invalidated": {}, "archives": {}}
    for name in ("scores.jsonl", "chinese.jsonl"):
        path = root / "predictions" / name
        pending = path.with_suffix(".reconciled.pending")
        stale = []
        try:
            with pending.open("w", encoding="utf-8", newline="\n") as stream:
                for row in read_rows(path):
                    if row["id"] in targets and row["text_sha256"] != targets[row["id"]]:
                        stale.append({**row, "superseded_reason": "target_text_changed_after_source_extension",
                                      "new_text_sha256": targets[row["id"]]})
                    else:
                        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            if stale:
                archive = root / "predictions/superseded" / f"{path.stem}-{digest[:16]}.jsonl"
                archive.parent.mkdir(exist_ok=True)
                if archive.exists():
                    raise ValueError("Refusing to overwrite superseded scores archive")
                archive.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in stale), encoding="utf-8")
                pending.replace(path)
                report["archives"][name] = str(archive.relative_to(root))
            report["invalidated"][name] = len(stale)
        finally:
            pending.unlink(missing_ok=True)
    atomic_json(root / "predictions/reconciliation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "data/acg/candidates.jsonl")
    args = parser.parse_args()
    print(json.dumps(reconcile(ROOT / "data/acg", args.input), ensure_ascii=False, indent=2), flush=True)
