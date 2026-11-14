from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from emotionlab.corpus.acg_cleaning import build
    print(json.dumps(build(ROOT / "data/acg"), ensure_ascii=False, indent=2), flush=True)
