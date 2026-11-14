"""Cache the independent second teacher used for ACG agreement filtering."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    from emotionlab.corpus.network import prepare_model_snapshot
    print(prepare_model_snapshot(ROOT / "models/acg-teacher", repo="Horizon-Labs/multilingual-emotions-base",
        revision="4f95ad28e89def7cce33b0589de4ed369b970deb", license_name="Apache-2.0"), flush=True)
