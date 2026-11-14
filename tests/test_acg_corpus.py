from collections import Counter
import hashlib
import json

import pytest

from emotionlab.common import LABELS
from emotionlab.corpus.acg_cleaning import clean, narrative_spans, NearIndex, paragraphs, split_for_work
from emotionlab.corpus.acg_annotation import decide_pair
from emotionlab.corpus.acg_verifier import decide_ensemble
from emotionlab.corpus.acg_publication import capped, CAPS, finalize_splits


class Identity:
    def convert(self, value):
        return value


def scores(label, high=.99):
    return {**dict.fromkeys(LABELS, .001), label: high}


def test_quotes_are_not_label_context_and_negation_is_preserved():
    result = list(narrative_spans('她愤怒地说：「我一点也不高兴！」然后转过身。'))
    assert result == [("narration", "她愤怒地说："), ("dialogue", "我一点也不高兴！"), ("narration", "然后转过身。")]
    assert clean(result[1][1], Identity())[0] == "我一点也不高兴!"
    assert clean("我恨你！", Identity())[0] == "我恨你!"
    assert clean("等等……我真的很害怕！", Identity())[0] is not None
    for text in ("加结构体系说明。", "第一章 新的起点", "请帮我扩写这句话。", "请访问https://example.com领取奖励"):
        assert clean(text, Identity())[0] is None


def test_lexical_near_variants_are_removed_before_splitting():
    index = NearIndex()
    original = "她推开了那扇破旧的大门，沿着长长的走廊一路走到了尽头。"
    variant = "她推开了那扇破旧的大门，沿着长长的走廊一路走到了尽处。"
    assert index.add(original, "first", "train") is None
    found = index.add(variant, "second", "test")
    assert found and found[:2] == ("first", "train")
    assert found[2] >= .85
    assert split_for_work("game-1") == split_for_work("game-1")
    assert {split_for_work(f"game-{i}") for i in range(100)} == {"train", "test", "validation"}


def test_review_rating_is_never_used_as_a_sentence_label(tmp_path):
    path = tmp_path / "reviews.json"
    path.write_text(json.dumps({"response": {"reviews": [{"language": "schinese", "recommendationid": "1",
        "review": "我非常喜欢这部作品。", "voted_up": False, "author": {"steamid": "123"}}]}}), encoding="utf-8")
    source = {"file": "reviews.json", "parser": "steam_reviews", "appid": 1, "work_id": "steam-1", "title": "VN"}
    rows = list(paragraphs(tmp_path, source))
    assert len(rows) == 1 and rows[0][1] == "我非常喜欢这部作品。"
    assert "label" not in rows[0][0] and "voted_up" not in rows[0][0]
    assert rows[0][0]["source_url"].endswith("/recommended/1/")


def test_consensus_rejects_disagreement_and_does_not_invent_missing_emotions():
    assert decide_pair(scores("恐惧"), scores("中性"))[0] is None
    pair = {"first": scores("恐惧"), "second": scores("中性"), "unmapped": .01, "truncated": False}
    chinese = {"scores": scores("中性"), "unmapped": .001, "truncated": False}
    assert decide_ensemble(pair, chinese) == ("中性", None, ["mmbert", "chinese"])
    # Chinese concerned/questioning outputs supply no fear/surprise vote.
    chinese = {"scores": dict.fromkeys(LABELS, .01), "unmapped": .94, "truncated": False}
    assert decide_ensemble(pair, chinese)[0] is None
    pair["second"] = scores("恐惧")
    assert decide_ensemble(pair, chinese) == ("恐惧", None, ["xlmr", "mmbert"])
    pair["second"]["恐惧"] = float("nan")
    assert decide_ensemble(pair, chinese)[0] is None


def test_caps_do_not_manufacture_or_duplicate_minority_rows():
    rows = [{"id": str(i), "label": "中性" if i < 990 else "悲伤", "work_id": f"work-{i%20}",
             "content_type": "light_novel_dialogue"} for i in range(1000)]
    kept, rejected = capped(rows, split="train")
    counts = Counter(r["label"] for r in kept)
    assert counts["悲伤"] == 10
    assert counts["中性"]/len(kept) <= CAPS["neutral_fraction"]
    assert len({r["id"] for r in kept}) == len(kept)
    assert rejected["neutral_cap"] > 0


def test_bounded_sampling_does_not_magnify_a_tiny_silver_class_without_limit():
    from emotionlab.training.worker import balanced_sampler
    from emotionlab.config import TrainConfig
    rows = [{"label": "中性"}]*9990 + [{"label": "恐惧"}]*10
    sampler = balanced_sampler(rows, 42, power=.5, max_ratio=10)
    assert float(sampler.weights.max()/sampler.weights.min()) <= 10
    picked = Counter(rows[i]["label"] for i in sampler)
    assert 40 < picked["恐惧"] < 180
    with pytest.raises(ValueError, match="sampling_power"):
        TrainConfig.from_dict({"sampling_power": 1.1})


def test_final_release_removes_cross_split_lexical_matches_after_caps():
    original = "她推开了那扇破旧的大门，沿着长长的走廊一路走到了尽头。"
    variant = "她推开了那扇破旧的大门，沿着长长的走廊一路走到了尽处。"
    def row(uid, text, label, work):
        return {"id": uid, "text": text, "label": label, "work_id": work,
                "content_type": "light_novel_dialogue"}
    splits = {"train": [row("a", original, "中性", "one"), row("b", "能再和你见面真是太开心了！", "喜悦", "one")],
              "test": [row("c", variant, "中性", "two"), row("d", "这本书真的让我感到非常幸福。", "喜悦", "two")]}
    filtered, rejected = finalize_splits(splits)
    assert rejected["release_order_lexical_near_duplicate"] == 1
    assert {r["id"] for rows in filtered.values() for r in rows} == {"a", "b", "d"}


def test_abstention_guards_do_not_require_a_positive_keyword_for_every_emotion():
    from emotionlab.corpus.acg_quality import guard
    assert guard("这一幕让我手脚冰凉，我只想尽快逃离这里。", "恐惧") is None
    assert guard("我根本不高兴，只觉得悲伤。", "悲伤") is None
    assert guard("她内心十分孤独，见到同伴又感动落泪。", "悲伤") == "conflicting_explicit_affect"
    assert guard("加结构体系说明。", "中性").startswith("content_")
    assert guard("的混合使用，只有他才能做到。", "中性") == "fragment_or_author_note"


def test_filtered_editor_updates_the_original_row_index(tmp_path):
    from emotionlab.bridge import DesktopAPI
    from emotionlab.dataset import save_samples
    path = tmp_path / "review.jsonl"
    save_samples(path, [{"id": str(i), "text": text, "label": label, "domain": "acg",
                         "label_method": "independent_teacher_consensus", "scores": {"xlmr": scores(label)}, "confidence": .99}
                        for i, (text, label) in enumerate([
        ("我们明天去魔法学院。", "中性"), ("终于又和你见面，真开心！", "喜悦"), ("想起她离去的背影，我很难过。", "悲伤")])])
    api = DesktopAPI(dataset_file=str(path))
    try:
        filtered = api.dataset_page(0, 25, "悲伤")["data"]
        assert filtered["indices"] == [2] and filtered["total"] == 1
        assert api.update_sample(filtered["indices"][0], "这次我们终于重逢了！", "喜悦")["ok"]
        rows = api.dataset_page()["data"]["rows"]
        assert rows[0]["label"] == "中性" and rows[2]["label"] == "喜悦"
        assert rows[2]["review_status"] == "user_reviewed"
        assert rows[2]["domain"] == "user_edited" and rows[2]["original_domain"] == "acg"
        assert rows[2]["original_source_text"] == "想起她离去的背影，我很难过。"
        assert "scores" not in rows[2] and rows[2]["confidence"] is None
        assert rows[2]["previous_teacher_annotation"]["label"] == "悲伤"
    finally:
        api._manager.close()
