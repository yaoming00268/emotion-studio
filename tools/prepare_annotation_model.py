"""Download a pinned independent teacher, never the student's 1000-row checkpoint."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from emotionlab.corpus.network import prepare_teacher
    print(prepare_teacher(ROOT / "models/annotation-teacher"), flush=True)
