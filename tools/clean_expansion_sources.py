from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from emotionlab.corpus.cleaning import build_candidates
    print(json.dumps(build_candidates(ROOT / "data/expanded"), ensure_ascii=False, indent=2))
