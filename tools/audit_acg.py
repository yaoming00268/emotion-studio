"""Independently verify the released files, lineage, diversity and split isolation."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotionlab.common import LABELS, atomic_json
from emotionlab.dataset import dataset_fingerprint, load_samples, validate_pair
from emotionlab.corpus.acg_cleaning import NearIndex, split_for_work
from emotionlab.corpus.cleaning import duplicate_key
from emotionlab.corpus.acg_verifier import decide_ensemble
from emotionlab.corpus.acg_quality import guard
from emotionlab.corpus.acg_publication import read_rows, calibration_rows
from tools.fetch_acg_sources import APPS


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit(root):
    started = time.perf_counter()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["acg_only"] is True and manifest["domain"] == "acg"
    sources = json.loads((root / "candidate-sources.json").read_text(encoding="utf-8"))["sources"]
    source_map = {r["id"]: r for r in sources}
    assert len(source_map) == len(sources)
    for name, digest in manifest["supporting_artifacts"].items():
        assert sha(root / name) == digest, name
    snapshot = manifest["pool"]["input_snapshot"]
    assert sha(root / "candidates.jsonl") == snapshot["candidates_sha256"]
    assert sha(root / "candidate-sources.json") == snapshot["sources_sha256"]
    cleaning = json.loads((root / "cleaning-report.json").read_text(encoding="utf-8"))
    assert cleaning["source_registry_sha256"] == snapshot["sources_sha256"]
    assert cleaning["statistics"]["accepted"] == manifest["pool"]["candidate_count"]
    for source in sources:
        assert source["kind"] in {"light_novel", "visual_novel", "acg_review", "acg_introduction"}
        assert source["acg_evidence"] and source["rights_evidence"]
        assert sha(root / source["file"]) == source["sha256"], source["id"]
        if source["parser"] == "steam_reviews":
            appid = source["appid"]
            assert appid in APPS
            game = json.loads((root / source["metadata_evidence"]).read_text(encoding="utf-8"))[str(appid)]["data"]
            assert game["type"] == "game" and game["name"] == source["title"]
            assert APPS[appid].casefold() in game["name"].casefold()
    print(f"Verified {len(sources)} raw source hashes and rights/domain metadata", flush=True)
    splits, all_keys, numeric_keys, ids = {}, set(), set(), set()
    for split, file in manifest["files"].items():
        path = root / file["file"]
        assert sha(path) == file["sha256"], path
        rows = load_samples(path)
        assert len(rows) == file["count"] and dataset_fingerprint(rows) == file["semantic_sha256"]
        for row in rows:
            assert row["id"] not in ids
            ids.add(row["id"])
            key = duplicate_key(row["text"])
            assert row["id"] == "acg-" + hashlib.sha256(key.encode()).hexdigest()[:24]
            assert key not in all_keys, row["id"]
            all_keys.add(key)
            numeric_key = re.sub(r"\d+", "#", key)
            if len(key) >= 16:
                assert numeric_key not in numeric_keys, row["id"]
            numeric_keys.add(numeric_key)
            assert row["domain"] == "acg" and row["source"] in source_map
            source = source_map[row["source"]]
            assert row["license"] == source["license"]
            assert split_for_work(row["work_id"]) == ("test" if split == "test_reference" else split)
            assert row["human_reviewed"] is False
            assert row["label_method"] in {"independent_teacher_consensus", "assistant_review", "assistant_blind_review"}
            if source["parser"] == "steam_reviews":
                assert row["content_type"] == "acg_review"
                assert row["source_url"].startswith("https://steamcommunity.com/profiles/")
                assert row["source_url"].endswith(f"/recommended/{source['appid']}/")
            for scores in row["scores"].values():
                assert set(scores) == set(LABELS)
                assert all(math.isfinite(value) and 0 <= value <= 1 for value in scores.values())
            if row["label_method"] == "independent_teacher_consensus":
                supporters = row["supporting_teachers"]
                assert len(set(supporters)) >= 2
                pair = {"first": row["scores"]["xlmr"], "second": row["scores"]["mmbert"],
                        "unmapped": row["teacher_unmapped"]["mmbert"], "truncated": row["teacher_truncated"]["pair"]}
                chinese = {"scores": row["scores"]["chinese"], "unmapped": row["teacher_unmapped"]["chinese"],
                           "truncated": row["teacher_truncated"]["chinese"]}
                label, reason, actual_supporters = decide_ensemble(pair, chinese)
                assert label == row["label"] and reason is None and actual_supporters == supporters
                assert guard(row["text"], label) is None
                assert row["confidence"] == min(row["scores"][name][row["label"]] for name in supporters)
                assert row["review_status"] in {"not_reviewed", "assistant_reviewed"}
            else:
                assert row["review_status"] == "assistant_reviewed"
        splits[split] = rows
        if split != "test_reference":
            assert Counter(row["label"] for row in rows)["中性"]/len(rows) <= manifest["caps"]["neutral_fraction"]
            assert sum(row["content_type"] == "acg_review" for row in rows)/len(rows) <= manifest["caps"]["review_fraction"]
        print(split, len(rows), dict(Counter(row["label"] for row in rows)), flush=True)
    assert len(splits["train"]) >= 100000
    assert set(Counter(row["label"] for row in splits["train"])) == set(LABELS)
    assert max(Counter(row["work_id"] for row in splits["train"]).values()) <= manifest["caps"]["train_per_work"]
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        validate_pair(splits[left], splits[right])
        for field in ("work_id", "document_id", "cluster_id", "author_group"):
            assert not {r[field] for r in splits[left] if r.get(field)} & {r[field] for r in splits[right] if r.get(field)}, field
    for left in ("train", "validation"):
        for field in ("work_id", "document_id", "cluster_id", "author_group"):
            assert not {r[field] for r in splits[left] if r.get(field)} & {r[field] for r in splits["test_reference"] if r.get(field)}, field
    for row in calibration_rows(root):
        assert duplicate_key(row["text"]) not in all_keys
    released = {r["id"]: r for rows in splits.values() for r in rows}
    lineage_checked = 0
    for candidate in read_rows(root / "candidates.jsonl"):
        if candidate["id"] in released:
            row = released[candidate["id"]]
            for field in ("text", "source", "source_url", "work_id", "location", "content_type", "document_id", "license", "author_group"):
                assert row.get(field) == candidate.get(field), (row["id"], field)
            lineage_checked += 1
    assert lineage_checked == len(released)
    print("Verified cleaned candidate lineage for every released sentence", flush=True)
    # Re-query an approximate lexical index over the actual release, including
    # all splits. A zero result is not a claim of exhaustive semantic isolation.
    index, near_cross, near_total = NearIndex(), [], 0
    for split, rows in splits.items():
        for row in rows:
            match = index.add(row["text"], row["id"], split)
            if match:
                near_total += 1
                if match[1] != split:
                    near_cross.append({"left": match[0], "right": row["id"], "similarity": match[2]})
        print("Lexical near check", split, "done", flush=True)
    assert not near_cross, f"Detected cross-split lexical near matches: {near_cross[:3]}"
    report = {"ok": True, "domain": "acg", "source_files_checked": len(sources),
        "counts": {key: len(value) for key, value in splits.items()},
        "whole_work_overlap": 0, "review_author_overlap": 0, "exact_or_numeric_overlap": 0,
        "approximate_cross_split_near_matches": len(near_cross), "approximate_within_split_near_matches": near_total,
        "calibration_overlap": 0, "all_source_hashes_verified": True, "all_release_hashes_verified": True,
        "candidate_lineage_checked": lineage_checked,
        "elapsed_seconds": round(time.perf_counter()-started, 2),
        "scope": "Hash/provenance, ACG allowlist, caps, teacher votes and lexical leakage audit; not a claim of human annotation accuracy or exhaustive semantic deduplication."}
    atomic_json(root / "audit-report.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(audit(ROOT / "data/acg"), ensure_ascii=False, indent=2), flush=True)
