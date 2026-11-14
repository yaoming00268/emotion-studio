"""Cache ACG works and documented, public review API pages for local research.

Never downloads commercial game scripts, credentials, profiles, or game assets.
Review authors retain their rights: the corpus must not be advertised as CC data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotionlab.common import atomic_json, now
from emotionlab.corpus.network import fetch

AUTHOR_REPO = "asd567557275/Traditional_Chinese_noval_authors_upload"
AUTHOR_REV = "cd752186d57951bee10a879e5c2b5f4085c44b4f"
SNOW_REV = "da61e50c157de0b337f2a514aea6400f0ab88b41"
# IDs are checked against the public store metadata before any review is used.
# Every title below is an authored VN, or an anime-style narrative game.
APPS = {
    324160: "CLANNAD", 1230140: "ATRI", 897220: "Summer Pockets",
    1144400: "千恋", 888790: "Sabbat", 1277930: "Riddle Joker",
    1829980: "星光咖啡馆", 1044620: "苍之彼方", 782250: "Hush Hush",
    1277940: "金辉恋曲", 333600: "NEKOPARA", 412830: "STEINS;GATE",
    825630: "STEINS;GATE 0", 316720: "planetarian", 635940: "Little Busters",
    430960: "露西", 326480: "If My Heart Had Wings",
    668630: "Tricolour Lovestory", 1049100: "泡沫", 1434480: "爱上火车",
    648100: "Raging Loop", 976390: "9-nine", 1033420: "9-nine",
    1142830: "9-nine", 1424660: "9-nine", 1567800: "星空列车",
    1811500: "LOOPERS", 1345740: "恋爱绮谭",
    1161870: "昙花", 1506980: "葬花", 2593370: "饿殍", 1669980: "火山",
    698780: "Doki Doki", 1002560: "茸雪", 939400: "LoveChoice",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def obtain(url, path, delay):
    """Use verified cache; throttle new requests and stop on persistent errors."""
    existed = path.is_file()
    receipt = fetch(url, path)
    if not existed:
        time.sleep(delay)
    return receipt


def collect(root, *, pages=25, delay=1.2, workers=2, negative_pages=0):
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    sources = []
    # Pin the author's own upload, preserving the original attribution/license.
    for local, remote in [("author-novels-README.md", "README.md"), ("author-novels-train.jsonl", "train.jsonl")]:
        if not (raw / local).exists():
            fetch(f"https://huggingface.co/datasets/{AUTHOR_REPO}/resolve/{AUTHOR_REV}/{remote}", raw / local)
    sources.append({"id": "author-novels", "file": "raw/author-novels-train.jsonl",
        "url": f"https://huggingface.co/datasets/{AUTHOR_REPO}/tree/{AUTHOR_REV}",
        "revision": AUTHOR_REV, "sha256": sha(raw / "author-novels-train.jsonl"),
        "parser": "author_novels", "kind": "light_novel", "license": "CC-BY-4.0",
        "author": "睡半夜怎麼三更", "rights_evidence": "raw/author-novels-README.md",
        "acg_evidence": "Author-provided titles/tags: light novel, fantasy, school, mecha, isekai, visual novel branches.",
        "redistribution": "Attribution required; see author's pinned card."})
    old = ROOT / "data/expanded/raw/snow-love.rpy"
    if not (raw / "snow-love.rpy").exists() and old.is_file():
        (raw / "snow-love.rpy").write_bytes(old.read_bytes())
    if (raw / "snow-love.rpy").is_file():
        sources.append({"id": "snow-love", "file": "raw/snow-love.rpy", "sha256": sha(raw / "snow-love.rpy"),
            "url": f"https://huggingface.co/datasets/Mosior/aquasofts/tree/{SNOW_REV}", "revision": SNOW_REV,
            "parser": "renpy", "kind": "visual_novel", "work_id": "snow-love", "title": "雪之恋",
            "license": "CC-BY-NC-SA-4.0", "author": "Aquasofts / Mosior",
            "rights_evidence": "https://huggingface.co/datasets/Mosior/aquasofts/blob/" + SNOW_REV + "/README.md",
            "acg_evidence": "Original Chinese RenPy visual novel; author's own upload."})
    for name, url in {
        "steam-api-doc.html": "https://partner.steamgames.com/doc/webapi/IUserReviewsService",
        "steam-api-terms.html": "https://steamcommunity.com/dev/apiterms",
        "steam-store-robots.txt": "https://store.steampowered.com/robots.txt",
    }.items():
        obtain(url, raw / name, delay)
    stats, completed = Counter(), {}
    fixed_sources = list(sources)

    def one_game(appid, expected, negative=False):
        game_sources = []
        store_path = raw / "steam" / str(appid) / "store.json"
        try:
            obtain("https://store.steampowered.com/api/appdetails?" + urlencode({"appids": appid, "l": "schinese"}), store_path, delay)
            info = json.loads(store_path.read_text(encoding="utf-8"))[str(appid)]
            if not info.get("success"):
                return [], 0, "unavailable_title", expected
            game = info["data"]
            title = game["name"]
            if expected.casefold() not in title.casefold():
                print("SKIP title mismatch", appid, expected, title, flush=True)
                return [], 0, "mismatched_title", title
            if game.get("type") != "game":
                return [], 0, "not_game", title
            cursor, seen, review_count = "*", set(), 0
            for page in range(negative_pages if negative else pages):
                if (root / "STOP_COLLECTION").exists():
                    break
                params = {"appid": appid, "filter": 1, "languages": ["schinese", "tchinese"],
                    "num_per_page": 100, "cursor": cursor, "purchase_type": 1,
                    "review_type": 2 if negative else 0, "filter_offtopic_activity": True}
                url = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/?" + urlencode({"input_json": json.dumps(params, separators=(",", ":"))})
                prefix = "negative-" if negative else ""
                path = raw / "steam" / str(appid) / f"{prefix}page-{page:04}.json"
                receipt = obtain(url, path, delay)
                body = json.loads(path.read_text(encoding="utf-8"))
                response = body.get("response", body)
                reviews = response.get("reviews", [])
                if not isinstance(reviews, list):
                    raise ValueError("Unexpected review response")
                if reviews:
                    game_sources.append({"id": f"steam-{appid}-{prefix}{page:04}", "file": str(path.relative_to(root)).replace("\\", "/"),
                        "sha256": receipt["sha256"], "fetched_at": receipt["fetched_at"], "url": url,
                        "parser": "steam_reviews", "kind": "acg_review", "appid": appid,
                        "work_id": f"steam-{appid}", "title": title, "author": "Individual public review authors (per-row permalink)",
                        "license": "Steam-Web-API-terms; author rights retained",
                        "rights_evidence": "raw/steam-api-terms.html", "metadata_evidence": str(store_path.relative_to(root)).replace("\\", "/"),
                        "acg_evidence": f"Curated VN/anime narrative game allowlist; public store title verified: {title}",
                        "review_sampling": "recent negative supplement; rating is not a sentence label" if negative else "recent all ratings",
                        "redistribution": "Local personal research cache. No open content license is claimed; review rights remain with authors."})
                review_count += len(reviews)
                next_cursor = response.get("cursor")
                if not reviews or not next_cursor or next_cursor == cursor or next_cursor in seen:
                    break
                seen.add(cursor)
                cursor = next_cursor
            return game_sources, review_count, "negative_games" if negative else "games", title
        except Exception as error:
            atomic_json(root / "collection-error.json", {"appid": appid, "error": str(error), "at": now()})
            # Do not evade authentication, bans, or rate limits through other hosts.
            raise
    with ThreadPoolExecutor(max_workers=workers) as pool:
        modes = (False, True) if negative_pages else (False,)
        futures = {pool.submit(one_game, appid, expected, negative): (appid, negative)
                   for negative in modes for appid, expected in APPS.items()}
        for future in as_completed(futures):
            appid, negative = futures[future]
            game_sources, review_count, state, title = future.result()
            completed[appid, negative] = game_sources
            stats["reviews"] += review_count
            stats[state] += 1
            # Deterministic allowlist order, regardless of network completion order.
            sources = fixed_sources + [s for mode in modes for key in APPS for s in completed.get((key, mode), [])]
            atomic_json(root / "sources.json", {"created_at": now(), "acg_only": True,
                "sources": sources, "collection": dict(stats), "review_limit_per_game": pages * 100,
                "review_order": "most recent all ratings, plus a separately recorded negative supplement when requested; ratings never substitute for emotion labels",
                "state": "completed" if len(completed) == len(futures) else "collecting", "workers": workers,
                "negative_review_limit_per_game": negative_pages * 100})
            print(json.dumps({"appid": appid, "title": title, "negative_supplement": negative,
                "reviews": review_count, "total": stats["reviews"]}, ensure_ascii=False), flush=True)
    return dict(stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data/acg")
    parser.add_argument("--pages", type=int, default=25)
    parser.add_argument("--delay", type=float, default=1.2)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--negative-pages", type=int, default=0)
    args = parser.parse_args()
    print(collect(args.root, pages=args.pages, delay=args.delay, workers=args.workers, negative_pages=args.negative_pages), flush=True)
