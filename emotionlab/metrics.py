"""Metrics from actual predictions, independent of the training implementation."""
import math


def classification_metrics(targets, predictions, mean_loss, labels):
    if len(targets) != len(predictions) or not targets:
        raise ValueError("指标需要数量相同、非空的人工标签和预测标签。")
    matrix = [[0 for _ in labels] for _ in labels]
    for target, predicted in zip(targets, predictions):
        if not 0 <= target < len(labels) or not 0 <= predicted < len(labels):
            raise ValueError("标签索引超出范围。")
        matrix[target][predicted] += 1
    details = []
    for index, label in enumerate(labels):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted_count = sum(row[index] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0.
        recall = tp / support if support else 0.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        details.append({"label": label, "precision": precision, "recall": recall, "f1": f1, "support": support})
    if not math.isfinite(mean_loss):
        raise ValueError("损失为非有限数值。")
    return {"loss": mean_loss, "accuracy": sum(a == b for a, b in zip(targets, predictions)) / len(targets),
            "macro_f1": sum(row["f1"] for row in details) / len(labels),
            "confusion_matrix": matrix, "per_label": details, "sample_count": len(targets)}
