"""Pure-data operations shared by annotation and training; no model imports."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from .common import LABELS, atomic_json, normalized_text


def load_samples(path, labels=LABELS, *, allow_unlabeled=False):
    path = Path(path)
    if path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError("标注文件超过 512MB，请拆分。")
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    elif path.suffix.lower() == ".json":
        with path.open(encoding="utf-8-sig") as stream:
            rows = json.load(stream)
    elif path.suffix.lower() == ".txt" and allow_unlabeled:
        from .common import sentences
        rows = [{"text": text, "label": ""} for text in sentences(path.read_text(encoding="utf-8-sig"))]
    else:
        with path.open(encoding="utf-8-sig") as stream:
            return _validate_rows((json.loads(line) for line in stream if line.strip()), labels, allow_unlabeled)
    if not isinstance(rows, list):
        raise ValueError("标注数据必须为样本列表。")
    return _validate_rows(rows, labels, allow_unlabeled)


def _validate_rows(rows, labels, allow_unlabeled):
    result = []
    for index, row in enumerate(rows, 1):
        if index > 500000:
            raise ValueError("单个标注文件最多支持 50 万句，请拆分。")
        if not isinstance(row, dict) or not isinstance(row.get("text"), str) or not normalized_text(row["text"]):
            raise ValueError(f"第 {index} 句缺少有效 text。")
        label = row.get("label", "")
        if label not in labels and not (allow_unlabeled and label == ""):
            raise ValueError(f"第 {index} 句标签无效或未标注：{label}")
        result.append({**row, "id": str(row.get("id", f"row-{index:04}")), "text": row["text"].strip(), "label": label})
    return result


def validate_pair(train, validation, labels=LABELS):
    if not train or not validation:
        raise ValueError("训练集和验证集都必须非空。")
    seen = {}
    for split, rows in (("train", train), ("validation", validation)):
        counts = Counter(row["label"] for row in rows)
        missing = set(labels) - set(counts)
        if missing:
            raise ValueError(f"{split} 缺少标签样本：" + "、".join(sorted(missing)))
        for row in rows:
            key = normalized_text(row["text"])
            if key in seen:
                origin, label = seen[key]
                if label != row["label"]:
                    raise ValueError("同句存在冲突标签：" + row["text"])
                raise ValueError(f"发现重复或训练/验证泄漏（{origin} / {split}）：{row['text']}")
            seen[key] = split, row["label"]
    return {"train_count": len(train), "validation_count": len(validation),
            "train_labels": dict(Counter(row["label"] for row in train)),
            "validation_labels": dict(Counter(row["label"] for row in validation))}


def save_samples(path, rows, labels=LABELS):
    # Write a separate temporary file; never leave a partially written dataset.
    import os
    import uuid
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex + ".jsonl")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        load_samples(temporary, labels, allow_unlabeled=True)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def dataset_fingerprint(rows):
    # Keep the v1 canonical representation while avoiding a huge intermediate string.
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(sort_keys=True, ensure_ascii=False)
    digest.update(b"[")
    for index, row in enumerate(rows):
        if index:
            digest.update(b", ")
        # v1 fingerprints cover identity/text/target only. Provenance is kept
        # in the file and independently hashed in corpus manifests.
        canonical = {"id": row["id"], "text": row["text"], "label": row["label"]}
        for part in encoder.iterencode(canonical):
            digest.update(part.encode("utf-8"))
    digest.update(b"]")
    return digest.hexdigest()


def audit_datasets(root):
    root = Path(root)
    train, validation = load_samples(root / "train.jsonl"), load_samples(root / "validation.jsonl")
    stats = validate_pair(train, validation)
    stats.update(train_sha256=dataset_fingerprint(train), validation_sha256=dataset_fingerprint(validation))
    return stats
