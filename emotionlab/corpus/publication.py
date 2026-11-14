"""Publish traceable silver labels; keep source chapters out of other splits."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from ..common import LABELS, atomic_json, now
from ..dataset import load_samples, save_samples, validate_pair, dataset_fingerprint
from .quality import assess, contract as quality_contract, VERSION as QUALITY_VERSION


def split_for(document_id):
    bucket = int(hashlib.sha256(("emotion-expansion-v1|" + document_id).encode()).hexdigest()[:8], 16) % 100
    return "test" if bucket < 5 else "validation" if bucket < 10 else "train"


def _annotations(root):
    annotations = {}
    with (root / "annotations.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in annotations:
                raise ValueError("Duplicate annotation ID: " + row["id"])
            annotations[row["id"]] = row
    return annotations


def prepare_review(root, *, per_label=20):
    root = Path(root)
    annotations = _annotations(root)
    groups = defaultdict(list)
    eligible = set()
    filtering = Counter()
    for line in (root / "candidates.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        label = annotations[row["id"]]["label"]
        if label is not None:
            reason, _ = assess(row["text"], label)
            if reason:
                filtering[reason] += 1
            else:
                groups[label].append(row["id"])
                eligible.add(row["id"])
    rng = random.Random(20261112)
    chosen = {}
    for label in LABELS:
        ids = groups[label]
        rng.shuffle(ids)
        chosen.update((uid, "stratified_random") for uid in ids[:per_label])
        # Additional near-threshold rows are diagnostic, not a random quality estimate.
        boundary = sorted(ids, key=lambda uid: (annotations[uid]["confidence"], uid))[:5]
        chosen.update((uid, "near_threshold") for uid in boundary if uid not in chosen)
    # Retain decisions already made in the pilot whenever the row survives
    # the new guards. Known pilot errors must not silently become unreviewed.
    pilot_file = root / "review/pilot-decisions.jsonl"
    if pilot_file.exists():
        for line in pilot_file.read_text(encoding="utf-8").splitlines():
            uid = json.loads(line)["id"]
            if uid in eligible:
                chosen.setdefault(uid, "pilot_retained")
    rows = []
    with (root / "candidates.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["id"] in chosen:
                annotation = annotations[row["id"]]
                rows.append({**row, **annotation, "selection": chosen[row["id"]],
                             "decision": None, "reviewer": None, "reason": ""})
    rows.sort(key=lambda row: (LABELS.index(row["label"]), row["selection"], row["id"]))
    audit = root / "review"
    audit.mkdir(exist_ok=True)
    save = audit / "sample-to-review.jsonl"
    with save.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic_json(audit / "sample-design.json", {"seed": 20261112, "strata": "teacher label after explicit-expression guards",
        "random_per_label": per_label, "additional_near_threshold_per_label": 5,
        "random_population": {label: len(groups[label]) for label in LABELS}, "sample_count": len(rows),
        "quality_filter": quality_contract(), "filter_rejections": dict(filtering),
        "pilot_retained": sum(row["selection"] == "pilot_retained" for row in rows),
        "created_at": now(), "reviewer_type": "coding assistant; not a human annotation panel"})
    return rows


def publish(root):
    root = Path(root)
    status = json.loads((root / "annotation-status.json").read_text(encoding="utf-8"))
    if status["state"] != "completed":
        raise ValueError("Complete annotation before publishing")
    annotations = _annotations(root)
    contract = json.loads((root / "annotation-contract.json").read_text(encoding="utf-8"))
    review_file = root / "review/assistant-decisions.jsonl"
    if not review_file.is_file():
        raise ValueError("Assistant review decisions must be recorded before publishing")
    reviews = {}
    for line in review_file.read_text(encoding="utf-8").splitlines():
        review = json.loads(line)
        if review["decision"] not in ("keep", "correct", "exclude") or review["reviewer"] != "codex":
            raise ValueError("Unfinished assistant review")
        if review["decision"] == "correct" and review.get("label") not in LABELS:
            raise ValueError("Invalid review target")
        if review["id"] in reviews or not review.get("reason", "").strip():
            raise ValueError("Duplicate review ID or missing review reason")
        reviews[review["id"]] = review
    sampled = [json.loads(line) for line in (root / "review/sample-to-review.jsonl").read_text(encoding="utf-8").splitlines()]
    if set(reviews) != {row["id"] for row in sampled}:
        raise ValueError("Review decisions must cover the exact review sample")
    splits = {key: [] for key in ("train", "validation", "test")}
    documents = {key: set() for key in splits}
    rejected, reviewed_counts = Counter(), Counter()
    missing, accepted = 0, 0
    for line in (root / "candidates.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        annotation = annotations.get(row["id"])
        if annotation is None:
            missing += 1
            continue
        if annotation["label"] is None:
            rejected[annotation["rejection"]] += 1
            continue
        reason, evidence = assess(row["text"], annotation["label"])
        if reason:
            rejected["quality_" + reason] += 1
            continue
        review = reviews.get(row["id"])
        if review:
            reviewed_counts[review["decision"]] += 1
            if review["decision"] == "exclude":
                rejected["assistant_review_exclusion"] += 1
                continue
        result = {**row, **{key: value for key, value in annotation.items() if key not in ("rejection",)},
            "teacher": contract["teacher"], "teacher_revision": contract["teacher_revision"],
            "review_status": "assistant_reviewed" if review else "not_reviewed",
            "human_reviewed": False, "quality_filter": QUALITY_VERSION, "teacher_emotion_evidence": evidence}
        if review:
            result.update(reviewer="codex", review_decision=review["decision"], review_reason=review["reason"])
            if review["decision"] == "correct":
                result.update(teacher_label=result["label"], teacher_confidence=result["confidence"], confidence=None,
                              label=review["label"], label_method="assistant_review")
        split = split_for(row["document_id"])
        splits[split].append(result)
        documents[split].add(row["document_id"])
        accepted += 1
    if missing or len(annotations) != status["processed"]:
        raise ValueError(f"Incomplete annotation coverage: missing={missing}")
    if len(annotations) != accepted + sum(rejected.values()):
        raise ValueError("Annotation IDs do not exactly match candidate IDs")
    if len(splits["train"]) < 100000:
        raise ValueError(f"Fewer than 100000 unique scraped training sentences: {len(splits['train'])}")
    original = load_samples(root.parent / "train.jsonl")
    for row in original:
        row.update(label_method="authored_synthetic", human_reviewed=False, genre="synthetic_startup")
    splits["train"].extend(original)
    # Shuffle is deterministic; held-out chapters remain wholly excluded from training.
    rng = random.Random(20261112)
    for rows in splits.values():
        rng.shuffle(rows)
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if documents[left] & documents[right]:
            raise ValueError("Source chapter leakage")
        validate_pair(splits[left], splits[right])
    validate_pair(splits["train"], load_samples(root.parent / "validation.jsonl"))
    files = {}
    for split, rows in splits.items():
        name = "train.jsonl" if split == "train" else split + ".silver.jsonl"
        save_samples(root / name, rows)
        with (root / name).open("rb") as stream:
            file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        files[split] = {"file": name, "count": len(rows), "sha256": file_hash,
            "semantic_sha256": dataset_fingerprint(rows), "bytes": (root/name).stat().st_size,
            "labels": dict(Counter(row["label"] for row in rows)),
            "sources": dict(Counter(row.get("source", "unknown") for row in rows)),
            "genres": dict(Counter(row.get("genre", "unknown") for row in rows)),
            "source_documents": len(documents[split])}
    random_reviews = [row for row in sampled if row["selection"] == "stratified_random"]
    by_label = {}
    for label in LABELS:
        ids = [row["id"] for row in random_reviews if row["label"] == label]
        by_label[label] = {"reviewed": len(ids), "kept": sum(reviews[uid]["decision"] == "keep" for uid in ids)}
    report = {"created_at": now(), "version": 1, "train_count": len(splits["train"]),
        "scraped_train_count": len(splits["train"])-len(original), "synthetic_startup_count": len(original),
        "train_labels": files["train"]["labels"], "files": files,
        "accepted_total": accepted, "rejected": dict(rejected), "teacher_contract": contract,
        "quality_filter": quality_contract(),
        "assistant_review": {"sample_size": len(reviews), "decisions": dict(Counter(row["decision"] for row in reviews.values())),
            "applied_after_quality_filters": dict(reviewed_counts), "random_by_label": by_label,
            "note": "Assistant review is not human gold. The stratified and boundary samples do not establish overall annotation accuracy."},
        "split_method": "source chapter SHA256 buckets 90/5/5, no chapter overlap, exact/numeric dedup before split",
        "default_validation": "../validation.jsonl (140 synthetic startup references, excluded from expansion)",
        "silver_evaluation_note": "Held-out silver sets measure agreement with the same teacher. They are not independent human gold evaluation.",
        "licenses": "Per-source licenses apply; Snow Love is CC-BY-NC-SA-4.0. Not a blanket CC0 dataset.",
        "quality_limits": ["domain shift from public-domain fiction", "class imbalance", "explicit-expression selection bias", "uncalibrated automatic scores",
                           "subjective single-label mapping", "semantic near duplicates can remain", "small assistant review sample"]}
    for name in ("sources.json", "cleaning-report.json", "annotation-contract.json", "review/assistant-decisions.jsonl",
                 "review/pilot-decisions.jsonl", "review/sample-design.json", "review/review-report.json"):
        artifact = root / name
        if artifact.exists():
            report.setdefault("supporting_artifacts", {})[name] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    report["assistant_review"]["unique_sentences_read"] = json.loads((root / "review/review-report.json").read_text(encoding="utf-8"))["unique_sentences_read"]
    atomic_json(root / "manifest.json", report)
    return report
