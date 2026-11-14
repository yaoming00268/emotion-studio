"""CUDA inference with an independent, pinned multi-label emotion teacher."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

from ..common import LABELS, atomic_json, now

TEACHER = "tabularisai/multilingual-emotion-classification"
REVISION = "b70346f3d6015955e4ae7fdd345358f0811190a7"
MAPPING = {
    "中性": ["neutral"], "喜悦": ["joy", "love", "gratitude"], "悲伤": ["sadness"],
    "愤怒": ["anger", "frustration"], "恐惧": ["fear"], "惊讶": ["surprise"],
    "厌恶": ["disgust", "contempt"],
}


def decide(scores, *, threshold=.8, neutral_threshold=.9, margin=.2):
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values()):
        return None, "non_finite_scores"
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)
    first, second = ordered[:2]
    required = neutral_threshold if first == "中性" else threshold
    if scores[first] < required:
        return None, "low_score"
    if scores[first] - scores[second] < margin:
        return None, "mixed_emotion"
    # A single-label student should not learn conflicting active teacher heads.
    if scores[second] >= .5:
        return None, "multiple_active_emotions"
    return first, None


class EmotionTeacher:
    def __init__(self, path, *, budget_mb=2048):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if not torch.cuda.is_available():
            raise RuntimeError("批量标注需要 CUDA。")
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(budget_mb * 1024**2 / total)
        torch.cuda.reset_peak_memory_stats()
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True,
            trust_remote_code=False, use_safetensors=True, dtype=torch.float16, attn_implementation="sdpa").to("cuda").eval()
        self.ids = {name: int(i) for i, name in self.model.config.id2label.items()}
        if set(self.ids) != {name for names in MAPPING.values() for name in names}:
            raise ValueError("Teacher label identities differ from the pinned mapping")
        if self.model.config.problem_type != "multi_label_classification":
            raise ValueError("Teacher must use independent sigmoid heads")

    def predict(self, texts, *, max_length=256):
        import torch
        encoded = self.tokenizer(texts, padding=True, truncation=False, return_length=True)
        encoded.pop("length")
        lengths = [sum(mask) for mask in encoded["attention_mask"]]
        encoded = {k: [row[:max_length] for row in v] for k, v in encoded.items()}
        inputs = {k: torch.tensor(v, dtype=torch.long, device="cuda") for k, v in encoded.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            logits = self.model(**inputs).logits
        heads = logits.float().sigmoid().cpu().tolist()
        return [(dict(zip(LABELS, [max(row[self.ids[name]] for name in MAPPING[label]) for label in LABELS])),
                 length > max_length) for row, length in zip(heads, lengths)]

    def memory(self):
        import torch
        return dict(peak_allocated_mib=round(torch.cuda.max_memory_allocated()/1024**2, 1),
                    peak_reserved_mib=round(torch.cuda.max_memory_reserved()/1024**2, 1))


def annotate(root, model_path, *, batch_size=16, limit=None):
    root = Path(root)
    candidates = root / "candidates.jsonl"
    with candidates.open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
    contract = {"teacher": TEACHER, "teacher_revision": REVISION, "input_sha256": fingerprint,
        "mapping": MAPPING, "threshold": .8, "neutral_threshold": .9, "margin": .2,
        "secondary_score_max": .5, "max_length": 256, "annotation_version": 1,
        "score_type": "uncalibrated independent sigmoid; merged labels use max, not softmax"}
    contract_file = root / "annotation-contract.json"
    if contract_file.exists() and json.loads(contract_file.read_text(encoding="utf-8")) != contract:
        raise ValueError("Annotation settings or candidate data changed. Choose a new output directory.")
    atomic_json(contract_file, contract)
    target = root / "annotations.jsonl"
    done = set()
    counts = Counter()
    if target.exists():
        # A crashed last partial record must be repaired explicitly, not ignored.
        with target.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                done.add(row["id"])
                counts[row.get("rejection") or row["label"]] += 1
    teacher = EmotionTeacher(model_path)
    started, processed = time.perf_counter(), 0
    buffer = []
    source_counts = Counter()
    status_file = root / "annotation-status.json"
    try:
        with target.open("a", encoding="utf-8", newline="\n") as stream:
            def flush():
                nonlocal processed
                if not buffer:
                    return
                predictions = teacher.predict([row["text"] for row in buffer])
                for row, (scores, truncated) in zip(buffer, predictions):
                    label, rejection = decide(scores)
                    if truncated:
                        label, rejection = None, "token_limit"
                    rounded = {name: round(value, 6) for name, value in scores.items()}
                    first = max(scores, key=scores.__getitem__)
                    annotation = {"id": row["id"], "label": label, "scores": rounded,
                        "confidence": round(scores[first], 6), "rejection": rejection,
                        "label_method": "independent_teacher", "human_reviewed": False}
                    stream.write(json.dumps(annotation, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counts[rejection or label] += 1
                    if label:
                        source_counts[row["source"]] += 1
                    processed += 1
                stream.flush()
                buffer.clear()
                if processed % 1024 == 0:
                    atomic_json(status_file, {"state": "annotating", "updated_at": now(), "processed": processed + len(done),
                        "counts": dict(counts), "elapsed_seconds": round(time.perf_counter()-started, 1), **teacher.memory()})
                    print(json.dumps({"processed": processed + len(done), "counts": dict(counts),
                        "sentences_per_second": round(processed/(time.perf_counter()-started), 1)}, ensure_ascii=False), flush=True)

            with candidates.open(encoding="utf-8") as incoming:
                for line in incoming:
                    if (root / "STOP_ANNOTATION").exists():
                        flush()
                        raise InterruptedError("Annotation stopped; rerun to resume")
                    row = json.loads(line)
                    if row["id"] in done:
                        continue
                    if limit is not None and processed + len(buffer) >= limit:
                        break
                    buffer.append(row)
                    if len(buffer) == batch_size:
                        flush()
                flush()
        state = {"state": "sampled" if limit else "completed", "updated_at": now(), "processed": processed+len(done),
            "counts": dict(counts), "elapsed_seconds": round(time.perf_counter()-started, 2), **teacher.memory()}
        atomic_json(status_file, state)
        return state
    except BaseException as error:
        atomic_json(status_file, {"state": "stopped" if isinstance(error, InterruptedError) else "failed",
                    "error": str(error), "processed": processed + len(done), "counts": dict(counts), "updated_at": now()})
        raise
