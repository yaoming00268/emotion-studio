"""Chinese-language cross-check, with concerned/questioning left unmapped."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from emotionlab.corpus.network import prepare_model_snapshot
    print(prepare_model_snapshot(ROOT / "models/chinese-verifier", repo="Johnson8187/Chinese-Emotion-Small",
        revision="a8b83db21e778e99d56058124eaab4ab99efe598", license_name="MIT"), flush=True)
