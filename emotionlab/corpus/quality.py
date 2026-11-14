"""Conservative acceptance guards motivated by a recorded pilot review.

These rules only reject teacher proposals; they never manufacture labels.
Lexical evidence reduces recall and adds an explicit-expression bias.
"""
from __future__ import annotations

import re

VERSION = "explicit-expression-v2"
CUES = {
    "喜悦": "高兴|开心|欣喜|喜悦|欢喜|大喜|喜不自胜|喜出望外|兴高采烈|欢天喜地|欢呼|欢乐|愉快|愉悦|欣慰|满意|欣然|感激|感谢|感恩|多谢|爱慕|喜爱|喜欢|庆幸|欢欣|笑逐颜开|眉开眼笑|喜笑颜开|欢欢喜喜|乐不可支|称心|欢悦",
    "悲伤": "悲伤|伤心|难过|悲哀|悲痛|伤悲|哀伤|哀痛|痛哭|哭泣|哭道|哭着|哭著|哭了|泪流|流泪|落泪|掉泪|挥泪|洒泪|涕泣|抽泣|啜泣|心碎|失落|绝望|沮丧|惆怅|惋惜|叹息|懊悔|哀哀|号哭|恸哭|哭诉|哭喊|哭声|痛心|忧愁|愁闷|可惜",
    "愤怒": "愤怒|大怒|怒道|怒曰|怒吼|怒骂|怒喝|怒视|发怒|恼怒|恼羞成怒|暴怒|气愤|气恼|生气|怒气|怒火|怨气|气急败坏|咬牙切齿|勃然|愤恨|愤愤|气死|恼了|大骂|喝骂|骂道|痛骂|斥骂|气得|气的|盛怒|气不打一处|忍无可忍|怒不可遏|火冒三丈|七窍生烟",
    "恐惧": "害怕|惧怕|畏惧|恐惧|惊恐|惊慌|惶恐|慌张|惊惶|惊骇|惊吓|担心|担忧|忧虑|不安|提心吊胆|心惊胆战|战战兢兢|毛骨悚然|胆战|胆颤|胆寒|吓得|吓的|吓坏|吓出|吓了一|心慌|忐忑|瑟瑟发抖|战栗|颤栗|恐慌|仓皇|心惊|慌了|惶惶",
    "惊讶": "惊讶|惊奇|惊诧|惊异|惊疑|诧异|诧道|惊愕|惊呆|吃惊|震惊|震撼|难以置信|不可思议|目瞪口呆|瞠目结舌|出乎意料|意想不到|闻所未闻|大吃一惊|惊问|惊道|惊曰",
    "厌恶": "厌恶|嫌恶|憎恶|憎恨|嫌弃|鄙视|鄙夷|轻蔑|不屑|唾弃|厌烦|厌弃|反感|看不起|瞧不起|恶心|可恶|讨厌|嫌憎|令人作呕|作呕|嗤之以鼻|嗤笑|冷笑|卑鄙|无耻|龌龊|下流|贱人|贱种|狗东西|畜生|混账|混帐|不要脸|无赖|愚蠢|愚昧|废话|恶毒",
}
PATTERNS = {label: re.compile(pattern) for label, pattern in CUES.items()}
NEGATED = re.compile(r"(?:不|没|无|未|毫无|没有|并非|不是)(?:再|很|太|曾|感到|觉得|十分|那么|怎么|一点|任何|什么|丝毫|值得|如此|有)?$")
BOILERPLATE = re.compile(r"且看下文|且听下回|且听下文|且看下回|欲知后事|有诗为证|诗曰|诗云|词曰|赞曰")
ENDING = re.compile(r"[。！？!?][’”」』\"）)]*$")
GARBLED_SCRIPT = re.compile(r"[\u0400-\u052f]")


def assess(text, label):
    """Return a rejection reason and the observed unnegated cue words."""
    if GARBLED_SCRIPT.search(text):
        return "garbled_foreign_script", {}
    if BOILERPLATE.search(text):
        return "narrative_boilerplate", {}
    if not ENDING.search(text):
        return "incomplete_sentence", {}
    evidence, negated = {}, set()
    for target, pattern in PATTERNS.items():
        matches = []
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start()-8):match.start()]
            if NEGATED.search(prefix) and not prefix.endswith("不胜"):
                negated.add(target)
            else:
                matches.append(match.group())
        if matches:
            evidence[target] = sorted(set(matches))
    if label == "中性":
        return ("neutral_with_emotion_cue", evidence) if evidence or negated else (None, evidence)
    if label in negated:
        return "negated_emotion_cue", evidence
    if label not in evidence:
        return "missing_explicit_emotion_cue", evidence
    if len(evidence) != 1:
        return "conflicting_emotion_cues", evidence
    return None, evidence


def contract():
    return {"version": VERSION, "cue_patterns": CUES, "negation_pattern": NEGATED.pattern,
            "garbled_script_pattern": GARBLED_SCRIPT.pattern,
            "boilerplate_pattern": BOILERPLATE.pattern, "ending_pattern": ENDING.pattern,
            "purpose": "Reject teacher proposals with missing, negated, or conflicting explicit emotional evidence.",
            "limits": "Heuristic rejection only, not verified gold. Biases data toward explicit expressions and neutral narration."}
