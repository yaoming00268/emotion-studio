from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from emotionlab.corpus import acg_publication as publication
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("calibrate", "pool", "review", "publish"))
    args = parser.parse_args()
    root = ROOT / "data/acg"
    result = {"calibrate": publication.calibrate, "pool": lambda r: publication.build_pool(r)[1],
              "review": publication.prepare_review, "publish": publication.publish}[args.action](root)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
