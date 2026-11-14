import json
from pathlib import Path

import pytest

from emotionlab.common import LABELS
from emotionlab.corpus.annotation import decide
from emotionlab.corpus.cleaning import clean_sentence, duplicate_key, renpy_utterance
from emotionlab.dataset import dataset_fingerprint, load_samples, save_samples
from emotionlab.corpus.publication import split_for
from emotionlab.corpus.quality import assess


class IdentityConverter:
    def convert(self, text):
        return text


def test_extract_renpy_without_executing_commands():
    assert renpy_utterance('m "谢谢学长！麻烦您了。"') == "谢谢学长！麻烦您了。"
    assert renpy_utterance('"我走到阳台边，推开窗户。"') == "我走到阳台边，推开窗户。"
    for line in ['play music "music.mp3"', 'define m = Character("木子米")',
                 '"去食堂吃饭":', '$ __import__("os").system("echo bad")', 'jump next_scene']:
        assert renpy_utterance(line) is None


def test_noise_and_punctuation_dedup_without_changing_emotion_negation():
    text, reason=clean_sentence('“{b}今天终于见到你了{/b}！”',IdentityConverter())
    assert text=="今天终于见到你了!" and reason is None
    assert duplicate_key(" 今天，终于见到你了！") == duplicate_key("今天终于见到你了!")
    assert duplicate_key("我并不开心") != duplicate_key("我很开心")
    for bad in ["请访问https://example.com获取更多内容", "第一章 欢乐的开始", "这是一句含有\ue007乱码的句子。"]:
        assert clean_sentence(bad,IdentityConverter())[0] is None


def test_multilabel_teacher_abstains_instead_of_inventing_neutral():
    low=dict.fromkeys(LABELS,.05)
    assert decide(low)==(None,"low_score")
    confident={**low,"喜悦":.98}
    assert decide(confident)==("喜悦",None)
    mixed={**confident,"悲伤":.65}
    assert decide(mixed)==(None,"multiple_active_emotions")
    assert decide({**low,"中性":.85})==(None,"low_score")
    assert decide({**low,"中性":.97})==("中性",None)
    assert decide({**low,"喜悦":float("nan")})==(None,"non_finite_scores")


def test_provenance_roundtrip_and_semantic_fingerprint(tmp_path):
    rows=[{"id":"sample-1","text":"我们终于又见面了！","label":"喜悦","source":"licensed-source",
           "source_url":"https://example.com/commit/story","confidence":.96,
           "label_method":"independent_teacher","human_reviewed":False}]
    path=tmp_path/"train.jsonl"
    save_samples(path,rows)
    result=load_samples(path)
    assert result==rows
    canonical=[{k:r[k] for k in ("id","text","label")} for r in rows]
    assert dataset_fingerprint(rows)==dataset_fingerprint(canonical)


def test_stream_jsonl_over_former_16mb_limit(tmp_path):
    # Large provenance is retained and no longer makes the dataset unreadable.
    row={"text":"今天我们一起出发。","label":"中性","provenance":"x"*(17*1024**2)}
    path=tmp_path/"large.jsonl"
    path.write_text(json.dumps(row,ensure_ascii=False)+"\n",encoding="utf-8")
    loaded=load_samples(path)
    assert loaded[0]["provenance"]==row["provenance"]


def test_split_is_stable_for_whole_source_chapters():
    splits={split_for(f"book:{i}") for i in range(1000)}
    assert splits=={"train","validation","test"}
    assert split_for("book:42")==split_for("book:42")


def test_quality_guard_abstains_on_missing_negated_or_conflicting_cues():
    assert assess("惧留孙说道:我来了!", "恐惧")[0] == "missing_explicit_emotion_cue"
    assert assess("这句话有字形Л乱码。", "中性")[0] == "garbled_foreign_script"
    assert assess("我并不高兴。", "喜悦")[0] == "negated_emotion_cue"
    assert assess("他大喜,却又悲伤地落泪。", "喜悦")[0] == "conflicting_emotion_cues"
    assert assess("且看下文分解。", "中性")[0] == "narrative_boilerplate"
    assert assess("今天终于结束", "中性")[0] == "incomplete_sentence"
    assert assess("我不胜感激!", "喜悦")[0] is None
    assert assess("他哀痛至切。", "悲伤")[0] is None


def test_balanced_sampler_corrects_skew_without_changing_unique_dataset():
    from emotionlab.training.worker import balanced_sampler
    from collections import Counter
    rows=[{"label":"中性"}]*990 + [{"label":"喜悦"}]*10
    indices=list(balanced_sampler(rows, 42))
    assert indices == list(balanced_sampler(rows, 42))
    assert len(indices)==len(rows) and min(indices)>=0 and max(indices)<len(rows)
    counts=Counter(rows[i]["label"] for i in indices)
    assert 400 < counts["喜悦"] < 600
    from emotionlab.config import TrainConfig
    with pytest.raises(ValueError, match="布尔"):
        TrainConfig.from_dict({"balance_classes":1})


def test_user_edit_cannot_reuse_teacher_score_for_a_different_target(tmp_path):
    from emotionlab.bridge import DesktopAPI
    path=tmp_path/"review.jsonl"
    save_samples(path,[{"id":"example","text":"这次旅程让我非常开心！","label":"喜悦",
        "label_method":"independent_teacher","scores":dict.fromkeys(LABELS,.1),"confidence":.99}])
    api=DesktopAPI(dataset_file=str(path))
    assert api.update_sample(0,"这次旅程让我非常难过。","悲伤")["ok"]
    row=api.dataset_page(0,1)["data"]["rows"][0]
    assert row["confidence"] is None and "scores" not in row
    assert row["previous_teacher_annotation"]["label"]=="喜悦"
    assert row["human_reviewed"] is True
    assert row["source_text_edited"] is True
