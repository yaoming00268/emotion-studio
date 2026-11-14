"""Download curated text only, with pinned revisions and attribution receipts."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emotionlab.common import atomic_json, now
from emotionlab.corpus.network import fetch

CLASSICS = "4a6d6f2088825f132521d848c2ea86cf9c9a7620"
NOVEL = "be27e745bf43ba34e18bd1f6fff1b42feaee1c40"
SNOW = "da61e50c157de0b337f2a514aea6400f0ab88b41"
BOOKS = [
    ("红楼梦", "集藏/小说/红楼梦.txt", "曹雪芹、高鹗"),
    ("西游记", "集藏/小说/西游记.txt", "吴承恩"),
    ("水浒传", "集藏/小说/水浒传.txt", "施耐庵、罗贯中"),
    ("儒林外史", "集藏/小说/儒林外史.txt", "吴敬梓"),
    ("二十年目睹之怪现状", "集藏/小说/二十年目睹之怪现状.txt", "吴趼人"),
    ("官场现形记", "集藏/小说/官场现行记.txt", "李宝嘉"),
    ("老残游记", "集藏/小说/老残游记.txt", "刘鹗"),
    ("儿女英雄传", "集藏/话本/儿女英雄传.txt", "文康"),
    ("三侠五义", "集藏/话本/三侠五义.txt", "石玉昆"),
    ("醒世恒言", "集藏/话本/醒世恒言.txt", "冯梦龙"),
    ("封神演义", "集藏/演义/封神演义.txt", "许仲琳"),
    ("三国演义", "集藏/演义/三国演义.txt", "罗贯中"),
]


def main():
    raw = ROOT / "data/expanded/raw"
    raw.mkdir(parents=True, exist_ok=True)
    sources = []

    def add(identifier, title, author, genre, license_name, url, parser, filename, revision, evidence):
        sources.append(dict(id=identifier, title=title, author=author, genre=genre, license=license_name,
            url=url, parser=parser, file=filename, revision=revision, license_evidence=evidence))

    novel_base = f"https://huggingface.co/datasets/asd567557275/chinese_novel/resolve/{NOVEL}/"
    add("space-grimoire", "空間魔導書與少年魔法師", "睡半夜怎麼三更", "light_novel", "CC-BY-4.0",
        novel_base + "chapters.jsonl", "chapters_jsonl", "space-grimoire.jsonl", NOVEL,
        "https://huggingface.co/datasets/asd567557275/chinese_novel/blob/" + NOVEL + "/README.md")
    snow_base = f"https://raw.githubusercontent.com/aquasofts/Snow-Love/{SNOW}/"
    add("snow-love", "雪之恋", "Mosior / aquasofts", "galgame", "CC-BY-NC-SA-4.0",
        snow_base + "game/script.rpy", "renpy", "snow-love.rpy", SNOW,
        f"https://github.com/aquasofts/Snow-Love/blob/{SNOW}/README.md")
    death_years = [1815, 1582, 1400, 1754, 1910, 1906, 1909, 1865, 1871, 1646, 1600, 1400]
    for i, (title, path, author) in enumerate(BOOKS, 1):
        add(f"classic-{i:02}", title, author, "public_domain_fiction", "Public-domain",
            f"https://raw.githubusercontent.com/garychowcmu/daizhigev20/{CLASSICS}/" + quote(path),
            "plain_novel", f"classic-{i:02}.txt", CLASSICS,
            f"https://github.com/garychowcmu/daizhigev20/blob/{CLASSICS}/" + quote("使用须知.md"))
        sources[-1].update(author_latest_death_year=death_years[i-1],
            rights_basis="Original work published before 1911; listed authors died before 1911. Repository notice is a source notice, not a blanket copyright license. Introductions and TOCs excluded.")
    evidence = [
        (novel_base + "README.md", "space-grimoire-LICENSE.md"),
        (snow_base + "README.md", "snow-love-LICENSE.md"),
        (f"https://raw.githubusercontent.com/garychowcmu/daizhigev20/{CLASSICS}/" + quote("使用须知.md"), "classics-NOTICE.md"),
    ]
    for url, name in evidence:
        fetch(url, raw / name)

    def download(source):
        receipt = fetch(source["url"], raw / source["file"])
        result = {**source, **receipt, "file": "raw/" + source["file"]}
        print(json.dumps({"source": source["id"], "bytes": receipt["bytes"]}, ensure_ascii=False), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(download, sources))
    atomic_json(ROOT / "data/expanded/sources.json", {"version": 1, "created_at": now(), "sources": results,
        "scope": "Licensed light novel + noncommercial Galgame script + public-domain fiction supplement; no paywalled scraping.",
        "changes": "Sentence extraction, traditional-to-simplified conversion, noise removal, deduplication and automatic emotion annotation."})


if __name__ == "__main__":
    main()
