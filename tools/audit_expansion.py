"""Read the published corpus through the exact training loader and audit splits."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from emotionlab.common import atomic_json, now
    from emotionlab.corpus.cleaning import duplicate_key
    from emotionlab.corpus.quality import assess
    from emotionlab.dataset import load_samples, validate_pair
    root = ROOT / "data/expanded"
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    sets = {name: load_samples(root/info["file"]) for name, info in manifest["files"].items()}
    seen, documents = {}, {}
    for name, rows in sets.items():
        info = manifest["files"][name]
        assert len(rows) == info["count"]
        assert dict(Counter(row["label"] for row in rows)) == info["labels"]
        documents[name] = set()
        for row in rows:
            key = duplicate_key(row["text"])
            assert key not in seen, (name, seen.get(key), row["id"])
            seen[key] = name
            if row["label_method"] != "authored_synthetic":
                assert row["source_url"] and row["location"] and row["document_id"]
                assert row["human_reviewed"] is False
                assert row["teacher_revision"] == manifest["teacher_contract"]["teacher_revision"]
                assert row["review_status"] in ("assistant_reviewed", "not_reviewed")
                assert set(row["scores"]) == set(manifest["train_labels"])
                assert all(math.isfinite(value) and 0 <= value <= 1 for value in row["scores"].values())
                assert row["quality_filter"] == manifest["quality_filter"]["version"]
                assert assess(row["text"], row.get("teacher_label", row["label"]))[0] is None
                if row["label_method"] == "assistant_review":
                    assert row["confidence"] is None and row["review_decision"] == "correct"
                else:
                    assert math.isfinite(row["confidence"]) and 0 <= row["confidence"] <= 1
                documents[name].add(row["document_id"])
        with (root/info["file"]).open("rb") as stream:
            assert hashlib.file_digest(stream,"sha256").hexdigest()==info["sha256"]
    for left,right in (("train","validation"),("train","test"),("validation","test")):
        assert not documents[left] & documents[right]
        validate_pair(sets[left],sets[right])
    seed_validation=load_samples(ROOT/"data/validation.jsonl")
    validate_pair(sets["train"],seed_validation)
    scraped=[row for row in sets["train"] if row["label_method"]!="authored_synthetic"]
    assert len(scraped)>=100000
    assert len({row["id"] for row in scraped})==len(scraped)
    for name, digest in manifest["supporting_artifacts"].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    with (root/"candidates.jsonl").open("rb") as stream:
        assert hashlib.file_digest(stream,"sha256").hexdigest() == manifest["teacher_contract"]["input_sha256"]
    sources=json.loads((root/"sources.json").read_text(encoding="utf-8"))["sources"]
    for source in sources:
        with (root/source["file"]).open("rb") as stream:
            assert hashlib.file_digest(stream,"sha256").hexdigest() == source["sha256"]
    report={"ok":True,"audited_at":now(),"train_count":len(sets["train"]),"scraped_train_count":len(scraped),
        "silver_validation_count":len(sets["validation"]),"silver_test_count":len(sets["test"]),
        "train_genres":dict(Counter(row["genre"] for row in sets["train"])),
        "checks":["training loader accepts full file", "at least 100000 unique scraped train rows", "valid seven-class labels",
                  "provenance and confidence retained", "no normalized exact duplicates", "no source chapter overlap",
                  "no original validation leakage", "file SHA256 verified", "post-teacher quality guards verified",
                  "raw source checksums and annotation input checksum verified", "review and source receipts verified"]}
    atomic_json(root/"verification.json",report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
