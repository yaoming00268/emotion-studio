"""Two independent emotion teachers; cached scores, explicit abstention.

Only the target sentence is classified. Context, Steam recommendations, and
work titles never become model inputs or substitute for sentence-level labels.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

from ..common import LABELS, atomic_json, now
from .annotation import EmotionTeacher, TEACHER, REVISION

SECOND_TEACHER = "Horizon-Labs/multilingual-emotions-base"
SECOND_REVISION = "4f95ad28e89def7cce33b0589de4ed369b970deb"
SECOND_MAPPING = {
    "中性": ["neutral"], "喜悦": ["joy", "amusement", "excitement", "love", "gratitude", "relief"],
    "悲伤": ["sadness", "grief", "disappointment", "remorse"], "愤怒": ["anger", "annoyance"],
    "恐惧": ["fear", "nervousness"], "惊讶": ["surprise"], "厌恶": ["disgust"],
}
DEFAULT_POLICY = {"first_threshold": .8, "second_threshold": .65, "first_neutral": .9,
    "second_neutral": .8, "margin": .2, "max_secondary": .5, "max_unmapped": .5, "max_length": 256}


def decide_pair(first, second, *, unmapped=0., truncated=False, policy=None):
    policy = policy or DEFAULT_POLICY
    values = [*first.values(), *second.values(), unmapped]
    if set(first) != set(LABELS) or set(second) != set(LABELS):
        return None, "label_schema"
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        return None, "invalid_scores"
    if truncated:
        return None, "token_limit"
    a, b = sorted(LABELS, key=first.__getitem__, reverse=True), sorted(LABELS, key=second.__getitem__, reverse=True)
    if a[0] != b[0]:
        return None, "teacher_disagreement"
    label = a[0]
    t1 = policy["first_neutral"] if label == "中性" else policy["first_threshold"]
    t2 = policy["second_neutral"] if label == "中性" else policy["second_threshold"]
    if first[label] < t1 or second[label] < t2:
        return None, "low_score"
    if first[label] - first[a[1]] < policy["margin"] or second[label] - second[b[1]] < policy["margin"]:
        return None, "small_margin"
    if first[a[1]] >= policy["max_secondary"] or second[b[1]] >= policy["max_secondary"]:
        return None, "mixed_emotions"
    if unmapped >= policy["max_unmapped"]:
        return None, "unmapped_emotion"
    return label, None


class SecondTeacher:
    def __init__(self, path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True,
            trust_remote_code=False, use_safetensors=True, dtype=torch.float16,
            attn_implementation="sdpa", reference_compile=False).to("cuda").eval()
        self.ids = {name: int(i) for i, name in self.model.config.id2label.items()}
        mapped = {name for names in SECOND_MAPPING.values() for name in names}
        if not mapped <= self.ids.keys() or len(self.ids) != 28:
            raise ValueError("Second teacher label identities differ from the pinned mapping")
        if self.model.config.problem_type != "multi_label_classification":
            raise ValueError("Second teacher must use independent sigmoid heads")
        self.unmapped = sorted(self.ids.keys() - mapped)

    def predict(self, texts, max_length=256):
        import torch
        encoded = self.tokenizer(texts, padding=True, truncation=False)
        lengths = [sum(mask) for mask in encoded["attention_mask"]]
        inputs = {key: torch.tensor([row[:max_length] for row in value], device="cuda") for key, value in encoded.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            heads = self.model(**inputs).logits.float().sigmoid().cpu().tolist()
        return [(dict(zip(LABELS, [max(row[self.ids[name]] for name in SECOND_MAPPING[label]) for label in LABELS])),
            max(row[self.ids[name]] for name in self.unmapped), length > max_length) for row, length in zip(heads, lengths)]


def annotate(root, first_path, second_path, *, batch_size=16, limit=None, input_file=None):
    import torch
    root = Path(root)
    cache = root / "predictions"
    cache.mkdir(parents=True, exist_ok=True)
    candidates = Path(input_file) if input_file else root / "candidates.jsonl"
    contract = {"first_teacher": TEACHER, "first_revision": REVISION, "second_teacher": SECOND_TEACHER,
        "second_revision": SECOND_REVISION, "second_mapping": SECOND_MAPPING, "max_length": 256,
        "version": 1, "score_type": "uncalibrated independent sigmoid; groups use maximum",
        "input": "target sentence only; never surrounding context or recommendation polarity"}
    contract_file = cache / "contract.json"
    if contract_file.is_file() and json.loads(contract_file.read_text(encoding="utf-8")) != contract:
        raise ValueError("Prediction cache has a different teacher contract")
    atomic_json(contract_file, contract)
    output = cache / "scores.jsonl"
    done = {}
    if output.exists():
        with output.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["id"] in done:
                    raise ValueError("Duplicate cached prediction id")
                done[row["id"]] = row["text_sha256"]
    first = EmotionTeacher(first_path)
    second = SecondTeacher(second_path)
    started, processed, reused = time.perf_counter(), 0, 0
    counts, buffer = Counter(), []
    with candidates.open("rb") as stream:
        input_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    status_path = root / "annotation-status.json"

    def status(state):
        return {"state": state, "processed_new": processed, "reused": reused, "counts": dict(counts),
            "input_file": str(candidates), "input_sha256": input_sha256, "updated_at": now(),
            "sentences_per_second": round(processed / max(.01, time.perf_counter() - started), 1), **first.memory()}

    try:
        with output.open("a", encoding="utf-8", newline="\n") as stream:
            def flush():
                nonlocal processed
                if not buffer:
                    return
                texts = [r["text"] for r in buffer]
                a, b = first.predict(texts), second.predict(texts)
                for row, (s1, trunc1), (s2, unmapped, trunc2) in zip(buffer, a, b):
                    label, rejection = decide_pair(s1, s2, unmapped=unmapped, truncated=trunc1 or trunc2)
                    result = {"id": row["id"], "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                        "first": {k: round(v, 6) for k, v in s1.items()}, "second": {k: round(v, 6) for k, v in s2.items()},
                        "unmapped": round(unmapped, 6), "truncated": trunc1 or trunc2}
                    stream.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counts[rejection or label] += 1
                    processed += 1
                stream.flush()
                buffer.clear()
                if processed % 1024 == 0:
                    atomic_json(status_path, status("annotating"))
                    print(json.dumps(status("annotating"), ensure_ascii=False), flush=True)
            with candidates.open(encoding="utf-8") as incoming:
                for line in incoming:
                    if (root / "STOP_ANNOTATION").exists():
                        flush()
                        raise InterruptedError("Stopped cooperatively; rerun to resume")
                    row = json.loads(line)
                    digest = hashlib.sha256(row["text"].encode()).hexdigest()
                    if row["id"] in done:
                        if digest != done[row["id"]]:
                            raise ValueError("Cached prediction target text changed")
                        reused += 1
                        continue
                    if limit is not None and processed + len(buffer) >= limit:
                        break
                    buffer.append(row)
                    if len(buffer) >= batch_size:
                        flush()
                flush()
        result = status("sampled" if limit else "completed")
        atomic_json(status_path, result)
        return result
    except BaseException as error:
        atomic_json(status_path, {**status("stopped" if isinstance(error, InterruptedError) else "failed"), "error": str(error)})
        raise
