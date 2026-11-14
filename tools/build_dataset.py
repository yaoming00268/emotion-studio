"""Materialize explicitly authored sentences; no template expansion or suffix padding."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotionlab.common import LABELS, atomic_json
from emotionlab.dataset import audit_datasets, save_samples


def main():
    sets = {"train": [], "validation": []}
    split = label = None
    for line in (ROOT / "data/corpus_source.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if match := re.fullmatch(r"\[(train|validation):(.+)\]", line):
            split, label = match.groups()
            if label not in LABELS:
                raise ValueError("Unknown label " + label)
            continue
        if split is None:
            raise ValueError("Missing split heading")
        sets[split].append({"id": f"{split}-{len(sets[split]) + 1:04}", "text": line, "label": label,
                            "source": "authored_synthetic"})
    counts = {name: len(rows) for name, rows in sets.items()}
    print(json.dumps({"counts": counts, "labels": {key: dict(Counter(row['label'] for row in rows)) for key, rows in sets.items()}}, ensure_ascii=False, indent=2))
    if counts != {"train": 1000, "validation": 140}:
        raise ValueError("The authored corpus must contain exactly 1000 train and 140 validation sentences.")
    for name, rows in sets.items():
        save_samples(ROOT / "data" / (name + ".jsonl"), rows)
    audit = audit_datasets(ROOT / "data")
    atomic_json(ROOT / "data/manifest.json", {"source": "authored_synthetic", "human_reviewed": False,
        "license": "CC0-1.0", "labels": LABELS, "notes": "原创合成启动语料；无模板批量扩写，未进行多标注者审校。验证集单独编写，不参与梯度更新。", **audit})


if __name__ == "__main__":
    main()
