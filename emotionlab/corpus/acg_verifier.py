"""A native Chinese tone classifier as a third, independently cached vote."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

from ..common import LABELS, atomic_json, now
from .acg_annotation import DEFAULT_POLICY, decide_pair

REPO = "Johnson8187/Chinese-Emotion-Small"
REVISION = "a8b83db21e778e99d56058124eaab4ab99efe598"
MAPPING = {0: "中性", 2: "喜悦", 3: "愤怒", 4: "悲伤", 6: "惊讶", 7: "厌恶"}
POLICY = {**DEFAULT_POLICY, "chinese_threshold": .85, "chinese_neutral": .9, "version": 2,
    "mmbert_neutral_veto": .8, "quality_guard_version": 1,
    "rule": "two independent confident votes; Chinese concerned/questioning abstain; fear requires both fear-capable teachers; strong mmBERT neutral/unmapped veto on other non-neutral two-vote decisions"}


def confident_vote(scores, threshold, neutral_threshold, *, margin=.2, secondary=.5, unmapped=0., unmapped_limit=.5):
    if set(scores) != set(LABELS) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in [*scores.values(), unmapped]):
        return None
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)
    first, second = ordered[:2]
    required = neutral_threshold if first == "中性" else threshold
    if (scores[first] < required or scores[first] - scores[second] < margin
            or scores[second] >= secondary or unmapped >= unmapped_limit):
        return None
    return first


def decide_ensemble(pair, chinese, *, policy=None):
    policy = policy or POLICY
    if pair["truncated"] or chinese["truncated"]:
        return None, "token_limit", []
    limits = {"margin": policy["margin"], "secondary": policy["max_secondary"], "unmapped_limit": policy["max_unmapped"]}
    a = confident_vote(pair["first"], policy["first_threshold"], policy["first_neutral"], **limits)
    b = confident_vote(pair["second"], policy["second_threshold"], policy["second_neutral"],
                       unmapped=pair["unmapped"], **limits)
    c = confident_vote(chinese["scores"], policy["chinese_threshold"], policy["chinese_neutral"],
                       unmapped=chinese["unmapped"], **limits)
    votes = [a, b, c]
    counts = Counter(x for x in votes if x is not None)
    if counts:
        label, count = counts.most_common(1)[0]
        if count >= 2 and (label != "恐惧" or (a == b == "恐惧")):
            if label != "中性" and b != label and (pair["unmapped"] >= policy["max_unmapped"] or pair["second"]["中性"] >= policy["mmbert_neutral_veto"]):
                return None, "verifier_contradiction", []
            return label, None, [name for name, value in zip(("xlmr", "mmbert", "chinese"), votes) if value == label]
    return None, "no_confident_consensus", [name + ":" + (value or "abstain") for name, value in zip(("xlmr", "mmbert", "chinese"), votes)]


class ChineseVerifier:
    def __init__(self, path):
        import os
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if not torch.cuda.is_available():
            raise RuntimeError("Chinese verification requires CUDA")
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        torch.cuda.set_per_process_memory_fraction(2048 * 1024**2 / torch.cuda.get_device_properties(0).total_memory)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True,
            trust_remote_code=False, use_safetensors=True, dtype=torch.float16).to("cuda").eval()
        if self.model.config.num_labels != 8:
            raise ValueError("Chinese verifier must have exactly the eight pinned tone labels")

    def predict(self, texts, max_length=256):
        import torch
        encoded = self.tokenizer(texts, padding=True, truncation=False)
        lengths = [sum(mask) for mask in encoded["attention_mask"]]
        inputs = {key: torch.tensor([r[:max_length] for r in value], device="cuda") for key, value in encoded.items()}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            heads = self.model(**inputs).logits.float().softmax(-1).cpu().tolist()
        result = []
        for scores, length in zip(heads, lengths):
            merged = dict.fromkeys(LABELS, 0.)
            merged.update({label: round(scores[index], 6) for index, label in MAPPING.items()})
            result.append({"scores": merged, "unmapped": round(scores[1]+scores[5], 6), "truncated": length > max_length})
        return result


def verify(root, model_path, *, input_file=None, batch_size=24):
    root = Path(root)
    source = Path(input_file) if input_file else root / "candidates.jsonl"
    output = root / "predictions/chinese.jsonl"
    output.parent.mkdir(exist_ok=True)
    contract = {"repo": REPO, "revision": REVISION, "mapping": MAPPING, "unmapped_ids": [1, 5],
        "unmapped_note": "Concerned and questioning are NEVER remapped to fear or surprise",
        "activation": "8-class softmax, as documented by the model author", "max_length": 256, "version": 1}
    contract_file = output.parent / "chinese-contract.json"
    # JSON turns integer keys into strings; compare the canonical encoding.
    encoded_contract = json.loads(json.dumps(contract))
    if contract_file.exists() and json.loads(contract_file.read_text(encoding="utf-8")) != encoded_contract:
        raise ValueError("Chinese prediction contract changed")
    atomic_json(contract_file, contract)
    done = {}
    if output.exists():
        for line in output.open(encoding="utf-8"):
            row = json.loads(line)
            if row["id"] in done:
                raise ValueError("Duplicate Chinese prediction id")
            done[row["id"]] = row["text_sha256"]
    model = ChineseVerifier(model_path)
    processed, reused, buffer, started = 0, 0, [], time.perf_counter()
    with source.open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
    status_file = root / "verification-status.json"

    def status(state):
        import torch
        return {"state": state, "processed_new": processed, "reused": reused, "input_sha256": fingerprint,
            "updated_at": now(), "sentences_per_second": round(processed / max(.01, time.perf_counter()-started), 1),
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated()/1024**2, 1)}

    try:
        with output.open("a", encoding="utf-8", newline="\n") as stream:
            def flush():
                nonlocal processed
                if not buffer:
                    return
                for row, result in zip(buffer, model.predict([r["text"] for r in buffer])):
                    result.update(id=row["id"], text_sha256=hashlib.sha256(row["text"].encode()).hexdigest())
                    stream.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                    processed += 1
                stream.flush()
                buffer.clear()
                if processed % 3072 == 0:
                    atomic_json(status_file, status("verifying"))
                    print(json.dumps(status("verifying"), ensure_ascii=False), flush=True)
            with source.open(encoding="utf-8") as incoming:
                for line in incoming:
                    if (root / "STOP_VERIFICATION").exists():
                        flush()
                        raise InterruptedError("Stopped cooperatively")
                    row = json.loads(line)
                    if row["id"] in done:
                        if done[row["id"]] != hashlib.sha256(row["text"].encode()).hexdigest():
                            raise ValueError("Cached Chinese target text changed")
                        reused += 1
                        continue
                    buffer.append(row)
                    if len(buffer) >= batch_size:
                        flush()
                flush()
        atomic_json(status_file, status("completed"))
        return status("completed")
    except BaseException as error:
        atomic_json(status_file, {**status("stopped" if isinstance(error, InterruptedError) else "failed"), "error": str(error)})
        raise
