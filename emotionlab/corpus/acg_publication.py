"""Publish traceable ACG silver data, with honest, separately reviewed checks."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from ..common import LABELS, atomic_json, now
from ..dataset import dataset_fingerprint, load_samples, save_samples, validate_pair
from .cleaning import duplicate_key
from .acg_verifier import POLICY, decide_ensemble
from .acg_quality import guard
from .acg_cleaning import NearIndex

CAPS = {"train_per_work": 20000, "neutral_fraction": .8, "review_fraction": .8, "introduction_fraction": .03}


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def cache_index(path):
    result = {}
    for row in read_rows(path):
        if row["id"] in result:
            raise ValueError("Duplicate cached id: " + row["id"])
        result[row["id"]] = row
    return result


def calibration_rows(root):
    return [r for name in ("calibration-decisions.jsonl", "calibration-emotion-decisions.jsonl")
            for r in read_rows(root / "review" / name)] + list(
                read_rows(root / "review/calibration-comments-decisions.jsonl")
                if (root / "review/calibration-comments-decisions.jsonl").is_file() else [])


def calibrate(root):
    """Report a frozen policy on actual assistant readings; never call it gold."""
    from .annotation import decide
    from .acg_annotation import decide_pair
    root = Path(root)
    pair, chinese = cache_index(root / "predictions/scores.jsonl"), cache_index(root / "predictions/chinese.jsonl")
    reviewed = calibration_rows(root)
    if len({r["id"] for r in reviewed}) != len(reviewed):
        raise ValueError("Duplicate calibration reading")
    for row in reviewed:
        digest = hashlib.sha256(row["text"].encode()).hexdigest()
        if pair[row["id"]]["text_sha256"] != digest or chinese[row["id"]]["text_sha256"] != digest:
            raise ValueError("Calibration prediction targets a different text")
        if row.get("reviewer") != "codex" or not row.get("reason") or row.get("assistant_label") not in [None, *LABELS]:
            raise ValueError("Incomplete calibration decision")
    result = {"created_at": now(), "reviewer": "codex", "human_gold": False,
        "sample_count": len(reviewed), "clearly_labeled": sum(r["assistant_label"] is not None for r in reviewed),
        "ambiguous_or_invalid": sum(r["assistant_label"] is None for r in reviewed), "policy": POLICY,
        "sampling": "96 content-type-stratified training sentences plus 70 stratified by first teacher's top label; additional training-only ACG comments stratified by accepted teacher class when available. All read with scores hidden; see individual sample designs.",
        "limitations": "Small nonrepresentative assistant reference. It cannot establish population accuracy; no held-out test was used to tune thresholds.",
        "comparison_scope": "old_single_teacher runs the old model's score thresholds only, without its former cue-word content filter; this is not a comparison to the complete archived pipeline.",
        "methods": {}}
    for method in ("old_single_teacher", "two_teachers", "three_teacher_consensus"):
        accepted, matched, ambiguous = Counter(), Counter(), Counter()
        for row in reviewed:
            a, c = pair[row["id"]], chinese[row["id"]]
            if method == "old_single_teacher":
                label, _ = decide(a["first"])
            elif method == "two_teachers":
                label, _ = decide_pair(a["first"], a["second"], unmapped=a["unmapped"], truncated=a["truncated"])
            else:
                label, _, _ = decide_ensemble(a, c)
                if label and guard(row["text"], label):
                    label = None
            if label:
                accepted[label] += 1
                matched[label] += label == row["assistant_label"]
                ambiguous[label] += row["assistant_label"] is None
        total, exact = sum(accepted.values()), sum(matched.values())
        result["methods"][method] = {"accepted": total, "matches_assistant": exact,
            "accepted_ambiguous_or_invalid": sum(ambiguous.values()),
            "match_fraction_of_accepted": round(exact/total, 6) if total else None,
            "coverage": round(total/len(reviewed), 6),
            "per_label": {label: {"accepted": accepted[label], "matches": matched[label], "ambiguous": ambiguous[label]} for label in LABELS}}
    atomic_json(root / "review/calibration-report.json", result)
    # Subsequent publication refuses silently changed policies.
    atomic_json(root / "label-policy.json", POLICY)
    return result


def capped(rows, *, split):
    """Deterministically retain diversity; never copy/augment to fill a quota."""
    rows = sorted(rows, key=lambda r: hashlib.sha256(("acg-retain-v1:" + r["id"]).encode()).digest())
    rejected, works, retained = Counter(), Counter(), []
    for row in rows:
        if split == "train" and works[row["work_id"]] >= CAPS["train_per_work"]:
            rejected["work_cap"] += 1
            continue
        works[row["work_id"]] += 1
        retained.append(row)
    # Recheck fractions until mutually dependent caps stop removing rows.
    while True:
        before = len(retained)
        for name, predicate, fraction in [
            ("neutral_cap", lambda r: r["label"] == "中性", CAPS["neutral_fraction"]),
            ("review_cap", lambda r: r["content_type"] == "acg_review", CAPS["review_fraction"]),
            ("introduction_cap", lambda r: r["content_type"] == "acg_introduction", CAPS["introduction_fraction"]),
        ]:
            rest = sum(not predicate(r) for r in retained)
            maximum = int(rest * fraction / (1-fraction))
            count, next_rows = 0, []
            for row in retained:
                if predicate(row):
                    count += 1
                    if count > maximum:
                        rejected[name] += 1
                        continue
                next_rows.append(row)
            retained = next_rows
        if len(retained) == before:
            break
    return retained, rejected


def finalize_splits(splits):
    """Recheck approximate lexical duplicates in actual release order and caps."""
    rejected = Counter()
    while True:
        before = sum(map(len, splits.values()))
        index = NearIndex()
        for split, rows in splits.items():
            rows, removed = capped(rows, split=split)
            rejected.update(removed)
            retained = []
            for row in rows:
                match = index.add(row["text"], row["id"], split)
                if match:
                    rejected["release_order_lexical_near_duplicate"] += 1
                else:
                    retained.append(row)
            splits[split] = retained
        if sum(map(len, splits.values())) == before:
            return splits, rejected


def build_pool(root):
    root = Path(root)
    policy = json.loads((root / "label-policy.json").read_text(encoding="utf-8"))
    if policy != POLICY:
        raise ValueError("Run calibration and freeze the current label policy first")
    pair = cache_index(root / "predictions/scores.jsonl")
    chinese = cache_index(root / "predictions/chinese.jsonl")
    reference_keys = {duplicate_key(r["text"]) for r in calibration_rows(root)}
    split_rows = {k: [] for k in ("train", "validation", "test")}
    counts, authors, candidate_ids = Counter(), defaultdict(set), set()
    for row in read_rows(root / "candidates.jsonl"):
        uid = row["id"]
        if uid in candidate_ids:
            raise ValueError("Duplicate candidate id")
        candidate_ids.add(uid)
        if row.get("domain") != "acg":
            raise ValueError("Non-ACG candidate")
        if uid not in pair or uid not in chinese:
            counts["missing_prediction"] += 1
            continue
        digest = hashlib.sha256(row["text"].encode()).hexdigest()
        if pair[uid]["text_sha256"] != digest or chinese[uid]["text_sha256"] != digest:
            raise ValueError("Prediction target changed")
        if duplicate_key(row["text"]) in reference_keys:
            counts["reserved_for_calibration"] += 1
            continue
        label, reason, supporters = decide_ensemble(pair[uid], chinese[uid])
        if label:
            reason = guard(row["text"], label)
        if reason:
            counts[reason] += 1
            continue
        # Keep three independent score vectors. Their minimum is not accuracy.
        score_vectors = {"xlmr": pair[uid]["first"], "mmbert": pair[uid]["second"], "chinese": chinese[uid]["scores"]}
        result = {**row, "label": label, "label_method": "independent_teacher_consensus",
            "scores": score_vectors, "supporting_teachers": supporters,
            "teacher_unmapped": {"mmbert": pair[uid]["unmapped"], "chinese": chinese[uid]["unmapped"]},
            "teacher_truncated": {"pair": pair[uid]["truncated"], "chinese": chinese[uid]["truncated"]},
            "confidence": min(score_vectors[name][label] for name in supporters),
            "score_type": "uncalibrated; minimum supporting teacher score, not correctness probability",
            "review_status": "not_reviewed", "human_reviewed": False}
        if row.get("author_group"):
            authors[row["author_group"]].add(row["split"])
        split_rows[row["split"]].append(result)
    if counts["missing_prediction"]:
        raise ValueError(f"Incomplete prediction coverage: {counts['missing_prediction']}")
    cross_authors = {author for author, splits in authors.items() if len(splits) > 1}
    for split, rows in split_rows.items():
        clean = [r for r in rows if r.get("author_group") not in cross_authors]
        counts["cross_split_review_author"] += len(rows) - len(clean)
        split_rows[split], rejected = capped(clean, split=split)
        counts.update(rejected)
    split_rows, rejected = finalize_splits(split_rows)
    counts.update(rejected)
    report = {"created_at": now(), "candidate_count": len(candidate_ids), "rejected": dict(counts),
        "input_snapshot": {"candidates_sha256": hashlib.sha256((root / "candidates.jsonl").read_bytes()).hexdigest(),
                           "sources_sha256": hashlib.sha256((root / "candidate-sources.json").read_bytes()).hexdigest()},
        "caps": CAPS, "counts": {k: len(v) for k, v in split_rows.items()},
        "labels": {k: dict(Counter(r["label"] for r in v)) for k, v in split_rows.items()},
        "content_types": {k: dict(Counter(r["content_type"] for r in v)) for k, v in split_rows.items()},
        "unique_works": {k: len({r['work_id'] for r in v}) for k, v in split_rows.items()},
        "semantic_sha256": {k: dataset_fingerprint(v) for k, v in split_rows.items()},
        "removed_cross_split_authors": len(cross_authors), "policy": POLICY}
    atomic_json(root / "pool-report.json", report)
    return split_rows, report


def prepare_review(root, *, per_label=20, gold_per_label=10):
    root = Path(root)
    splits, report = build_pool(root)
    if len(splits["train"]) < 100000:
        raise ValueError("Collect more unique ACG data before freezing the final review sample")
    rng, selected = random.Random(2026111303), []
    for label in LABELS:
        group = [r for r in splits["train"] if r["label"] == label]
        selected.extend({**r, "selection": "train_label_stratified_random"} for r in rng.sample(group, min(per_label, len(group))))
    for name, rows in [("sample-to-review.jsonl", selected)]:
        (root / "review" / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    gold = []
    for label in LABELS:
        group = [r for r in splits["test"] if r["label"] == label]
        for row in rng.sample(group, min(gold_per_label, max(0, len(group)-1))):
            gold.append({key: row[key] for key in ("id", "text", "work_id", "content_type", "source_url", "location")})
    rng.shuffle(gold)
    (root / "review/test-blind.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in gold), encoding="utf-8")
    atomic_json(root / "review/sample-design.json", {"seed": 2026111303,
        "train_review": "20 per retained teacher class, simple random within class; after all diversity caps",
        "train_review_count": len(selected), "blind_test": "Up to 10 per retained teacher class, leaving at least one silver test row per class; target labels/scores hidden from reader",
        "blind_test_count": len(gold), "test_is_population_representative": False,
        "human_gold": False, "reviewer": "codex", "pool": report})
    return {"train_review_count": len(selected), "blind_test_count": len(gold), **report}


def publish(root):
    root = Path(root)
    splits, pool = build_pool(root)
    design = json.loads((root / "review/sample-design.json").read_text(encoding="utf-8"))
    if (design["pool"].get("semantic_sha256") != pool["semantic_sha256"]
            or design["pool"].get("input_snapshot") != pool["input_snapshot"]):
        raise ValueError("The pool changed after review selection; freeze a new review sample first")
    sample = list(read_rows(root / "review/sample-to-review.jsonl"))
    reviews = cache_index(root / "review/assistant-decisions.jsonl")
    if set(reviews) != {r["id"] for r in sample}:
        raise ValueError("Review must cover the exact frozen training sample")
    for row in reviews.values():
        if row.get("reviewer") != "codex" or row.get("decision") not in {"keep", "correct", "exclude"} or not row.get("reason"):
            raise ValueError("Unfinished review")
        if row["decision"] == "correct" and row.get("label") not in LABELS:
            raise ValueError("Invalid corrected label")
    gold_sample = list(read_rows(root / "review/test-blind.jsonl"))
    gold_decisions = cache_index(root / "review/test-decisions.jsonl")
    if set(gold_decisions) != {r["id"] for r in gold_sample}:
        raise ValueError("Blind test review must be complete")
    test_reference, review_decisions, test_results = [], Counter(), Counter()
    test_per_label = {label: Counter() for label in LABELS}
    for split, rows in splits.items():
        retained = []
        for row in rows:
            if row["id"] in gold_decisions:
                decision = gold_decisions[row["id"]]
                label = decision.get("assistant_label")
                if decision.get("reviewer") != "codex" or not decision.get("reason"):
                    raise ValueError("Unfinished blind test decision")
                test_results["reviewed"] += 1
                test_results["matches_silver"] += label == row["label"]
                test_per_label[row["label"]]["reviewed"] += 1
                test_per_label[row["label"]]["matches_assistant"] += label == row["label"]
                test_per_label[row["label"]]["ambiguous_or_invalid"] += label is None
                if label is not None:
                    if label not in LABELS:
                        raise ValueError("Invalid blind test label")
                    reference = {**row, "teacher_label": row["label"], "label": label, "confidence": None,
                        "label_method": "assistant_blind_review", "review_status": "assistant_reviewed",
                        "review_reason": decision["reason"], "reference_type": "small_assistant_reference_not_human_gold",
                        "teacher_supporting_teachers": row["supporting_teachers"]}
                    reference.pop("supporting_teachers", None)
                    test_reference.append(reference)
                else:
                    test_results["ambiguous_or_invalid"] += 1
                continue
            review = reviews.get(row["id"])
            if review:
                review_decisions[review["decision"]] += 1
                if review["decision"] == "exclude":
                    continue
                row = {**row, "review_status": "assistant_reviewed", "reviewer": "codex", "review_decision": review["decision"],
                    "review_reason": review["reason"]}
                if review["decision"] == "correct":
                    row.update(teacher_label=row["label"], label=review["label"], confidence=None, label_method="assistant_review",
                               teacher_supporting_teachers=row["supporting_teachers"])
                    row.pop("supporting_teachers", None)
            retained.append(row)
        splits[split] = retained
    # A review correction/exclusion must not leave the published caps violated.
    splits, final_caps = finalize_splits(splits)
    if len(splits["train"]) < 100000:
        raise ValueError(f"Only {len(splits['train'])} unique ACG training sentences pass quality controls; 100000 required. Do not pad.")
    if set(r["label"] for r in splits["train"]) != set(LABELS):
        raise ValueError("The ACG training release must cover every configured emotion label")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        validate_pair(splits[left], splits[right])
        for key in ("work_id", "document_id", "cluster_id", "author_group"):
            if {r[key] for r in splits[left] if r.get(key)} & {r[key] for r in splits[right] if r.get(key)}:
                raise ValueError("Cross-split leakage: " + key)
    files = {}
    for split, rows in {**splits, "test_reference": test_reference}.items():
        path = root / (split + ".jsonl")
        save_samples(path, rows)
        with path.open("rb") as stream:
            fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
        files[split] = {"file": path.name, "count": len(rows), "sha256": fingerprint,
            "semantic_sha256": dataset_fingerprint(rows), "bytes": path.stat().st_size,
            "labels": dict(Counter(r["label"] for r in rows)),
            "content_types": dict(Counter(r["content_type"] for r in rows)),
            "works": dict(Counter(r["work_id"] for r in rows)),
            "reviewed": sum(r["review_status"] == "assistant_reviewed" for r in rows)}
    manifest = {"created_at": now(), "version": 1, "domain": "acg", "acg_only": True,
        "train_count": len(splits["train"]), "train_labels": files["train"]["labels"], "files": files,
        "synthetic_startup_count": 0, "public_domain_fiction_count": 0, "pool": pool, "final_cap_exclusions": dict(final_caps),
        "caps": CAPS, "policy": POLICY,
        "split_method": "Whole work/franchise SHA256 80/10/10 buckets; remove exact/numeric/lexical near duplicates; exclude all cross-split review authors",
        "evaluation_type": "validation/test are silver; separate small blind assistant reference; none is human gold",
        "assistant_review": {"calibration_read": len(calibration_rows(root)), "training_read": len(reviews),
            "training_decisions": dict(review_decisions), "blind_test": dict(test_results),
            "blind_test_per_teacher_label": {k: dict(v) for k, v in test_per_label.items()}, "human_gold": False},
        "license_note": "Per-source rights apply. Novels CC-BY-4.0; Snow Love CC-BY-NC-SA-4.0. Public reviews are a local personal research cache under Steam API terms, with authors' rights retained; no blanket open-content license.",
        "quality_limits": ["Uncalibrated model scores", "Assistant reference is small and class-stratified",
            "Teacher errors can be correlated; agreement does not establish accuracy",
            "Lexical approximate near-dedup is not semantic equivalence", "Rare classes remain limited", "Most novel text comes from one author",
            "Confidence filtering favors explicit emotion and can miss implicit affect; silver metrics measure teacher agreement"],
        "supporting_artifacts": {}}
    for name in ("candidate-sources.json", "cleaning-report.json", "label-policy.json", "predictions/contract.json", "predictions/chinese-contract.json",
                 "review/calibration-report.json", "review/calibration-decisions.jsonl", "review/calibration-emotion-decisions.jsonl",
                 "review/sample-design.json", "review/sample-to-review.jsonl", "review/test-blind.jsonl",
                 "review/assistant-decisions.jsonl", "review/test-decisions.jsonl"):
        manifest["supporting_artifacts"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    registry = json.loads((root / "candidate-sources.json").read_text(encoding="utf-8"))
    for source in registry["sources"]:
        for field in ("rights_evidence", "metadata_evidence"):
            evidence = source.get(field, "")
            if evidence and not evidence.startswith("https://"):
                path = (root / evidence).resolve()
                if not path.is_relative_to(root.resolve()) or not path.is_file():
                    raise ValueError("Missing local source evidence: " + evidence)
                if evidence not in manifest["supporting_artifacts"]:
                    manifest["supporting_artifacts"][evidence] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("review/calibration-comments-design.json", "review/calibration-comments-blind.jsonl", "review/calibration-comments-decisions.jsonl", "predictions/reconciliation.json"):
        if (root / name).is_file():
            manifest["supporting_artifacts"][name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    atomic_json(root / "manifest.json", manifest)
    return manifest
