"""Deterministic sentence extraction; never expand templates to meet a quota."""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import re
import unicodedata

from ..common import atomic_json, now

CHAPTER = re.compile(r"^\s*第\s*[零〇一二三四五六七八九十百千万两\d\s]+[回章节卷集部篇]")
SENTENCE = re.compile(r"[^。！？!?\n]+(?:[。！？!?]+[’”」』\"）)]*|$)")
REN_SAY = re.compile(r'^(?:[A-Za-z_]\w*(?:\s+[A-Za-z_]\w*)*\s+)?("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')(?:\s*(?:#.*|with\s+\w+))?$')
NOISE = re.compile(r"https?://|www\.|QQ群|加群|点击.{0,8}(收藏|阅读)|版权所有|转载.{0,6}注明|扫码|下载地址|(?:^|\s)(?:label|scene|jump|return|define|image|play|stop|show|hide)\s|[\ufffd\x00]")
EXPLICIT = re.compile(r"性交|交媾|阴茎|阴道|精液|阴部|肉棒|鸡巴|强奸|轮奸|淫交")


def duplicate_key(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return "".join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith(("P", "S")))


def clean_sentence(text, converter):
    text = converter.convert(html.unescape(unicodedata.normalize("NFKC", text)))
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    text = re.sub(r"\{/?[a-zA-Z][^{}]*\}", "", text)
    text = re.sub(r"<[^>]{1,100}>", "", text)
    text = re.sub(r"\s+", " ", text).strip(' \t“”「」『』\"')
    # Stage directions at the beginning are not part of a spoken sentence.
    text = re.sub(r"^\((?:环顾|看着|望着|感叹|微笑|叹气|小声|低声|大喊)[^)]{0,18}\)", "", text).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    if not 6 <= len(text) <= 180:
        return None, "length"
    if EXPLICIT.search(text):
        return None, "explicit_content"
    if NOISE.search(text) or CHAPTER.match(text) or re.search(r"\[[A-Za-z_]\w*\]", text):
        return None, "noise"
    if re.search(r"[\ue000-\uf8ff\u2500-\u257f]", text) or any(unicodedata.category(c) == "Cc" for c in text):
        return None, "garbled_control"
    chinese = len(re.findall(r"[\u3400-\u9fff]", text))
    if chinese < 4 or chinese / len(text) < .55:
        return None, "non_chinese"
    if re.search(r"(.)\1{5,}", text) or re.search(r"(?:插图|本章完|全文完|待续|正文|目录|作者[:：])", text):
        return None, "boilerplate"
    return text, None


def renpy_utterance(line):
    stripped = line.strip()
    # Only say statements and narrator literals; no Python execution or menus.
    match = REN_SAY.fullmatch(stripped)
    if not match or stripped.startswith(("define ", "image ", "play ", "queue ", "scene ", "voice ", "show ", "hide ")):
        return None
    try:
        value = ast.literal_eval(match[1])
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, str) else None


def paragraphs(path, source):
    if source["parser"] == "chapters_jsonl":
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            row = json.loads(line)
            document = str(row["chapter"])
            for i, paragraph in enumerate(row["text"].splitlines(), 1):
                yield document, f"paragraph:{i}", paragraph
    elif source["parser"] == "renpy":
        document = "start"
        for i, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if match := re.match(r"\s*label\s+(\w+)\s*:", line):
                document = match[1]
            value = renpy_utterance(line)
            if value:
                yield document, f"line:{i}", value.replace("\\n", "\n")
    else:
        # Public-domain editions can contain modern prefaces and TOCs. Drop
        # all introductory material before the first chapter and title lines.
        document, started = "intro", False
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        beginnings = [i for i, line in enumerate(lines) if re.match(r"^\s*第\s*(?:一|0*1)\s*[回章节]", line)]
        # A repeated first chapter heading separates the TOC/preface from text.
        body_start = beginnings[-1] if beginnings else 0
        for i, line in enumerate(lines, 1):
            if i - 1 < body_start:
                continue
            line = line.strip()
            if match := CHAPTER.match(line):
                number = re.sub(r"\s+", "", match[0])
                document = number
                started = True
                continue
            if started and line and re.search(r"[。！？!?]", line):
                yield document, f"line:{i}", line


class Deduplicator:
    def __init__(self, excluded=()):
        self.exact = {duplicate_key(text) for text in excluded}
        self.near = set()

    def add(self, text):
        key = duplicate_key(text)
        if key in self.exact:
            return "exact_duplicate"
        # Collapse digit-only variants as well, without altering saved text.
        masked = re.sub(r"\d+", "#", key)
        if len(key) >= 16 and masked in self.near:
            return "numeric_variant"
        self.exact.add(key)
        self.near.add(masked)
        return None


def build_candidates(root):
    from opencc import OpenCC
    converter = OpenCC("t2s")
    root = Path(root)
    sources = json.loads((root / "sources.json").read_text(encoding="utf-8"))["sources"]
    excluded = []
    for name in ("train", "validation"):
        for line in (root.parent / (name + ".jsonl")).read_text(encoding="utf-8").splitlines():
            excluded.append(json.loads(line)["text"])
    dedup = Deduplicator(excluded)
    stats, counts = Counter(), Counter()
    output = root / "candidates.jsonl"
    temporary = output.with_suffix(".pending")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for source in sources:
            path = root / source["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
                raise ValueError("Source changed after acquisition: " + source["id"])
            for document, location, paragraph in paragraphs(path, source):
                for index, match in enumerate(SENTENCE.finditer(paragraph), 1):
                    stats["raw_sentences"] += 1
                    text, reason = clean_sentence(match[0], converter)
                    if text:
                        reason = dedup.add(text)
                    if reason:
                        stats["rejected_" + reason] += 1
                        continue
                    uid = hashlib.sha256(duplicate_key(text).encode("utf-8")).hexdigest()[:24]
                    row = {"id": "web-" + uid, "text": text, "source": source["id"],
                        "source_url": source["url"], "genre": source["genre"], "license": source["license"],
                        "document_id": source["id"] + ":" + document,
                        "location": location + f":sentence:{index}", "human_reviewed": False}
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counts[source["id"]] += 1
                    stats["accepted"] += 1
            print(source["id"], counts[source["id"]], flush=True)
    temporary.replace(output)
    report = {"created_at": now(), "cleaning_version": 1, "statistics": dict(stats), "by_source": dict(counts),
        "steps": ["pinned source checksums", "RenPy dialogue extraction", "NFKC", "OpenCC t2s", "real sentence boundaries",
                  "6-180 characters", "Chinese ratio >=0.55", "remove code/ads/headings", "punctuation-insensitive dedup",
                  "numeric variant dedup", "remove private-use/control/box-drawing garbling", "remove explicit sexual vocabulary",
                  "exclude original train and validation"],
        "near_duplicate_limit": "Exact normalized and numeric variants removed; semantic paraphrases are not automatically equivalent."}
    atomic_json(root / "cleaning-report.json", report)
    return report
