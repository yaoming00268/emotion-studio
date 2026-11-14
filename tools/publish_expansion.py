from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from emotionlab.corpus.publication import prepare_review, publish
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare-review", action="store_true")
    args=parser.parse_args()
    root=ROOT/"data/expanded"
    if args.prepare_review:
        rows=prepare_review(root)
        print(json.dumps({"review_sample_count":len(rows)},ensure_ascii=False))
    else:
        print(json.dumps(publish(root),ensure_ascii=False,indent=2))
