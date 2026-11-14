"""ACG-only extraction with source lineage and conservative lexical deduplication."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import re
import unicodedata

from ..common import atomic_json, now
from .cleaning import duplicate_key, renpy_utterance

VERSION = 1
BOUNDARY = re.compile(r"[^。！？!?；;\n]+(?:[。！？!?；;]+[’”」』\"）)]*|$)")
QUOTES = re.compile(r"「([^「」]+)」|“([^“”]+)”")
NOISE = re.compile(r"https?://|www\.|QQ群|加群|扫码|下载地址|提取码|点个赞|奖励这条|求.{0,4}(点赞|点数)|购买头像|版权所有|章节目录|作者后记|译者后记|本章完|全文完|未完待续|加结构体系说明|(?:请|帮我).{0,6}(?:续写|扩写|润色|生成剧本)|[\ufffd\x00]")
EXPLICIT = re.compile(r"性交|交媾|阴茎|阴道|精液|肉棒|鸡巴|强奸|轮奸|淫交")
HEADING = re.compile(r"^\s*(?:第[零〇一二三四五六七八九十百千万两\d\s]+[回章节卷部篇]|(?:选项|选择|结局|分支|END|BE|HE)\s*[:：\d]|[A-DＡ-Ｄ]\s*[.、:：])")
FAMILIES = {825630: "steam-412830", 1033420: "steam-976390", 1142830: "steam-976390",
            1424660: "steam-976390", 1434480: "steam-466790"}


def split_for_work(work_id):
    value = int(hashlib.sha256(("acg-work-v1:" + work_id).encode()).hexdigest()[:8], 16) % 100
    return "test" if value < 10 else "validation" if value < 20 else "train"


def clean(text, converter):
    text = converter.convert(html.unescape(unicodedata.normalize("NFKC", text)))
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    text = re.sub(r"\{/?[a-zA-Z][^{}]*\}", "", text)
    text = re.sub(r"<[^>]{1,100}>", "", text)
    text = re.sub(r"\[/?(?:b|i|u|h[1-6]|spoiler|quote|list|olist|strike|code|url|img|table|tr|td)(?:=[^\]]*)?\]", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(' \t“”「」『』\"')
    if not 4 <= len(text) <= 180:
        return None, "length"
    if NOISE.search(text) or HEADING.match(text):
        return None, "noise_or_heading"
    if EXPLICIT.search(text):
        return None, "explicit_content"
    if re.search(r"[\ue000-\uf8ff\u2500-\u28ff]", text) or any(unicodedata.category(c) == "Cc" for c in text):
        return None, "garbled_control"
    chinese = len(re.findall(r"[\u3400-\u9fff]", text))
    if chinese < 3 or chinese / len(text) < .5:
        return None, "non_chinese"
    # Preserve normal ellipses, exclamations and laughter. Reject repeated art.
    if re.search(r"([\u3400-\u9fff])\1{9,}", text) or re.search(r"(.{3,16})\1{3,}", text):
        return None, "spam_repetition"
    if re.search(r"(?:说道|说着|问道|回答|表示|询问|开口|回答道)[:：]?$", text) or text.endswith(("：", ":", "，", ",", "、")):
        return None, "attribution_fragment"
    if len(text) < 8 and not re.search(r"[。！？!?]", text):
        return None, "short_fragment"
    if re.search(r"\[[A-Za-z_][^\]]*\]|\$[A-Za-z_]|(?:label|jump|scene)\s+[A-Za-z_]", text):
        return None, "script_placeholder"
    return text, None


def narrative_spans(paragraph):
    """Separate speech from narration without assigning surrounding emotions."""
    offset = 0
    for match in QUOTES.finditer(paragraph):
        prefix = paragraph[offset:match.start()].strip()
        if prefix:
            yield "narration", prefix
        yield "dialogue", match[1] or match[2]
        offset = match.end()
    if paragraph[offset:].strip():
        yield "narration", paragraph[offset:]


def paragraphs(root, source):
    path = root / source["file"]
    if source["parser"] == "author_novels":
        with path.open(encoding="utf-8-sig") as stream:
            for work in map(json.loads, stream):
                title = work["title"]
                # Same work across all chapters, editions, and future sources.
                wid = "novel-" + hashlib.sha256(title.encode()).hexdigest()[:16]
                for lineno, paragraph in enumerate(work["text"].splitlines(), 1):
                    for kind, span in narrative_spans(paragraph):
                        yield {"work_id": wid, "title": title, "author": work["author"],
                            "content_type": "light_novel_" + kind, "document_id": wid,
                            "location": f"line:{lineno}", "context": paragraph[:600]}, span
    elif source["parser"] == "renpy":
        scene = "start"
        for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            if match := re.match(r"\s*label\s+(\w+)\s*:", line):
                scene = match[1]
            value = renpy_utterance(line)
            if value:
                yield {"work_id": source["work_id"], "title": source["title"], "author": source["author"],
                    "content_type": "visual_novel_dialogue", "document_id": source["work_id"] + ":" + scene,
                    "location": f"line:{lineno}", "context": value[:600]}, value.replace("\\n", "\n")
    elif source["parser"] == "steam_reviews":
        body = json.loads(path.read_text(encoding="utf-8"))
        reviews = body.get("response", body).get("reviews", [])
        for review in reviews:
            if review.get("language") not in {"schinese", "tchinese"}:
                continue
            text = review.get("review", "")
            # Review-level spam rejection: do not keep an advert's first line.
            if NOISE.search(text) or len(re.findall(r"[\u2800-\u28ff]", text)) > 8:
                continue
            author = review.get("author", {}).get("steamid", "")
            appid, rid = source["appid"], str(review["recommendationid"])
            wid = FAMILIES.get(appid, source["work_id"])
            for lineno, line in enumerate(text.splitlines(), 1):
                yield {"work_id": wid, "title": source["title"], "content_type": "acg_review",
                    "document_id": "steam-review-" + rid, "location": f"review:{rid}:line:{lineno}",
                    "author": "public-review-" + hashlib.sha256(author.encode()).hexdigest()[:16],
                    "author_group": hashlib.sha256(author.encode()).hexdigest()[:16],
                    "source_url": f"https://steamcommunity.com/profiles/{author}/recommended/{appid}/",
                    "context": line[:600]}, line
    else:
        raise ValueError("Unapproved ACG parser: " + source["parser"])


def shingles(text):
    key = duplicate_key(text)
    return {key[i:i+4] for i in range(max(1, len(key)-3))}


class NearIndex:
    """LSH candidates plus exact Jaccard verification; never a semantic claim."""
    def __init__(self):
        self.buckets = defaultdict(list)
        self.rows = []

    def add(self, text, uid, split):
        key = duplicate_key(text)
        if len(key) < 16:
            return None
        grams = shingles(text)
        hashes = sorted(int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little") for g in grams)
        # Bottom-k sketches: bounded buckets avoid pathological boilerplate.
        sketch = hashes[:6]
        candidates = set()
        for h in sketch:
            candidates.update(self.buckets[h][-80:])
        for index in sorted(candidates):
            other, other_uid, other_split = self.rows[index]
            similarity = len(grams & other) / len(grams | other)
            if similarity >= .85:
                return other_uid, other_split, round(similarity, 5)
        index = len(self.rows)
        self.rows.append((grams, uid, split))
        for h in sketch:
            self.buckets[h].append(index)
        return None


def build(root):
    from opencc import OpenCC
    root = Path(root)
    registry_bytes = (root / "sources.json").read_bytes()
    registry = json.loads(registry_bytes)
    if registry.get("acg_only") is not True:
        raise ValueError("ACG source allowlist is required")
    converter, seen, numeric, near = OpenCC("t2s"), set(), set(), NearIndex()
    stats, kinds, works, sources = Counter(), Counter(), Counter(), Counter()
    output = root / "candidates.jsonl"
    pending = output.with_suffix(".pending")
    removed = root / "dedup-rejections.jsonl"
    with pending.open("w", encoding="utf-8", newline="\n") as stream, removed.open("w", encoding="utf-8", newline="\n") as duplicates:
        for source in registry["sources"]:
            if source["kind"] not in {"light_novel", "visual_novel", "acg_review", "acg_introduction"} or not source.get("acg_evidence"):
                raise ValueError("Missing ACG domain evidence: " + source["id"])
            path = root / source["file"]
            with path.open("rb") as raw:
                if hashlib.file_digest(raw, "sha256").hexdigest() != source["sha256"]:
                    raise ValueError("Source checksum changed: " + source["id"])
            for metadata, paragraph in paragraphs(root, source):
                for sentence_index, match in enumerate(BOUNDARY.finditer(paragraph)):
                    stats["raw_sentences"] += 1
                    text, reason = clean(match[0], converter)
                    if reason:
                        stats["rejected_" + reason] += 1
                        continue
                    key = duplicate_key(text)
                    uid = "acg-" + hashlib.sha256(key.encode()).hexdigest()[:24]
                    masked = re.sub(r"\d+", "#", key)
                    if key in seen or (len(key) >= 16 and masked in numeric):
                        stats["rejected_exact_or_numeric_duplicate"] += 1
                        continue
                    split = split_for_work(metadata["work_id"])
                    close = near.add(text, uid, split)
                    if close:
                        stats["rejected_near_duplicate"] += 1
                        duplicates.write(json.dumps({"id": uid, "representative": close[0], "similarity": close[2],
                            "work_id": metadata["work_id"], "source": source["id"]}, ensure_ascii=False) + "\n")
                        # Remove, rather than move a sentence out of its work's split.
                        continue
                    seen.add(key)
                    numeric.add(masked)
                    row = {"id": uid, "text": text, "source": source["id"], "source_url": source["url"],
                        "license": source["license"], "domain": "acg", "split": split,
                        "human_reviewed": False, "review_status": "not_reviewed", **metadata}
                    row["location"] += f":sentence:{sentence_index}"
                    row["cluster_id"] = uid
                    stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    stats["accepted"] += 1
                    kinds[row["content_type"]] += 1
                    works[row["work_id"]] += 1
                    sources[source["id"]] += 1
            if sum(sources.values()) % 10000 < 100:
                print(source["id"], stats["accepted"], flush=True)
    pending.replace(output)
    (root / "candidate-sources.json").write_bytes(registry_bytes)
    report = {"created_at": now(), "cleaning_version": VERSION, "statistics": dict(stats),
        "source_registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "by_content_type": dict(kinds), "by_work": dict(works), "by_source": dict(sources),
        "steps": ["ACG allowlist and raw SHA256 verification", "split speech from narration", "OpenCC t2s and NFKC",
            "strip markup without deleting negation or normal ellipses", "remove ads, placeholders, headings, explicit terms, spam",
            "punctuation-insensitive exact and numeric variant dedup", "bottom-6 char-4gram LSH with verified Jaccard >=0.85",
            "hold out complete works/franchises; never randomly split sentences"],
        "limits": "Near search is approximate lexical retrieval, not exhaustive or semantic deduplication. Context is review-only; teachers and student see only target text."}
    atomic_json(root / "cleaning-report.json", report)
    return report
