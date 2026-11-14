"""Abstain on broken targets and conflicting affect, without cue-only labels."""
from __future__ import annotations

import re

from .acg_cleaning import clean

CUES = {
    "喜悦": "开心 高兴 喜悦 欢乐 幸福 欣喜 兴奋 感激 感动 含笑 欢笑".split(),
    "悲伤": "悲伤 悲痛 伤心 难过 绝望 孤独 失落 痛哭 落泪 哭泣 哀声".split(),
    "愤怒": "愤怒 恼怒 气愤 怨恨 憎恨 仇恨 斥责".split(),
    "恐惧": "恐惧 害怕 惊恐 惊惧 吓得 担惊受怕 战栗".split(),
    "惊讶": "惊讶 震惊 错愕 惊诧 大吃一惊 惊掉下巴".split(),
    "厌恶": "恶心 厌恶 厌烦 鄙夷 鄙视 反感".split(),
}


class Identity:
    def convert(self, value):
        return value


def guard(text, label):
    _, reason = clean(text, Identity())
    if reason:
        return "content_" + reason
    if re.match(r"^的(?!确|士)", text) or re.match(r"^提到.{1,25}时[:：]", text):
        return "fragment_or_author_note"
    if label == "中性":
        if len(re.findall(r"[\u3400-\u9fff]", text)) < 8:
            return "short_context_dependent_neutral"
        if re.search(r"含笑|强作镇定|故作镇定|止不住的眼泪|哀声叹气", text):
            return "neutral_with_affect_expression"
    active = set()
    for emotion, terms in CUES.items():
        for term in terms:
            for match in re.finditer(re.escape(term), text):
                prefix = text[max(0, match.start()-5):match.start()]
                if not re.search(r"(?:不|没|未|无|并非|毫无|不会|没有|不是|不再)(?:那么|这么|太|很|再)?$", prefix):
                    active.add(emotion)
    # This is an abstention guard, never a requirement for an explicit cue.
    if len(active) >= 2:
        return "conflicting_explicit_affect"
    return None
