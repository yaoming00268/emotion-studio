from pathlib import Path
import argparse
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from emotionlab.corpus.annotation import annotate
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(annotate(ROOT / "data/expanded", ROOT / "models/annotation-teacher",
        batch_size=args.batch_size, limit=args.limit), ensure_ascii=False, indent=2))
