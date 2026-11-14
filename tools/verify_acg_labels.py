from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from emotionlab.corpus.acg_verifier import verify
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(verify(ROOT / "data/acg", ROOT / "models/chinese-verifier", input_file=args.input,
                           batch_size=args.batch_size), ensure_ascii=False, indent=2), flush=True)
