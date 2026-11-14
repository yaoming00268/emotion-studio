"""Audit teacher behavior on the existing synthetic reference set and source samples."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from emotionlab.common import atomic_json, LABELS
    from emotionlab.corpus.annotation import EmotionTeacher, decide
    from emotionlab.dataset import load_samples
    from emotionlab.metrics import classification_metrics
    teacher = EmotionTeacher(ROOT / "models/annotation-teacher")
    gold = load_samples(ROOT / "data/validation.jsonl")
    predictions, decisions = [], []
    for start in range(0, len(gold), 16):
        for row, (scores, truncated) in zip(gold[start:start+16], teacher.predict([x["text"] for x in gold[start:start+16]])):
            best = max(scores, key=scores.__getitem__)
            label, rejection = decide(scores)
            predictions.append(LABELS.index(best))
            decisions.append({"text": row["text"], "reference": row["label"], "top_label": best,
                              "accepted": label, "rejection": rejection, "scores": scores})
    metrics = classification_metrics([LABELS.index(x["label"]) for x in gold], predictions, 0., LABELS)
    metrics.pop("loss")
    report = {"reference_type": "140 synthetic startup sentences, not an independent human gold benchmark",
        "top1_metrics": metrics, "decisions": decisions, "memory": teacher.memory()}
    atomic_json(ROOT / "data/expanded/teacher-reference-check.json", report)
    print(json.dumps({"reference_type": report["reference_type"], "top1_accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"], "accepted": sum(x["accepted"] is not None for x in decisions),
        "mistakes": [{k: x[k] for k in ["text", "reference", "top_label", "accepted"]} for x in decisions if x["reference"]!=x["top_label"]],
        "memory": report["memory"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
