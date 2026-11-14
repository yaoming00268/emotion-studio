from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
import unicodedata
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = ["中性", "喜悦", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶"]
BASE_MODEL_ID = "google-bert/bert-base-chinese"
BASE_REVISION = "8f23c25b06e129b6c986331a13d8d025a92cf0ea"


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(4):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 3:
                    raise
                time.sleep(.02 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def normalized_text(text):
    # Ignore presentation-only differences when checking leakage/duplicates.
    return re.sub(r"[\s，。！？、；：,.!?;:‘’“”\"']+", "", unicodedata.normalize("NFKC", text).casefold())


def sentences(text):
    return [part.strip() for part in re.split(r"(?<=[。！？!?；;])|[\r\n]+|(?<=\.)\s+", text) if part.strip()]


def project_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
