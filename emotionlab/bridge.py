"""pywebview's desktop bridge: annotation, orchestration and analysis client only."""
from __future__ import annotations

from collections import Counter
from functools import wraps
import inspect
import json
from pathlib import Path
import subprocess

from .common import LABELS, PROJECT_ROOT, atomic_json, project_path, read_json, sentences
from .config import TrainConfig
from .dataset import load_samples, save_samples
from .processes import ProcessManager


def guarded(method):
    @wraps(method)
    def invoke(*args, **kwargs):
        try:
            return {"ok": True, "data": method(*args, **kwargs)}
        except Exception as error:
            return {"ok": False, "error": str(error)}
    invoke.__signature__ = inspect.signature(method)
    return invoke


class DesktopAPI:
    def __init__(self, dataset_file=None):
        self._manager = ProcessManager()
        self._window = None
        settings = read_json(PROJECT_ROOT / "runtime/settings.json", TrainConfig().to_dict())
        self._dataset_file = project_path(dataset_file or settings["train_file"])
        self._samples = load_samples(self._dataset_file, allow_unlabeled=True) if self._dataset_file.is_file() else []

    @guarded
    def bootstrap(self):
        settings = read_json(PROJECT_ROOT / "runtime/settings.json", TrainConfig().to_dict())
        return {"config": settings, "labels": LABELS, "project": str(PROJECT_ROOT), "runs": self._runs(),
                "base_ready": (project_path(settings["base_model"]) / "model.safetensors").is_file(),
                "service_file": str(self._manager.service_file), "dataset": self._dataset_stats(),
                "expansion": self._expansion_info()}

    def _expansion_info(self):
        manifest = read_json(PROJECT_ROOT / "data/acg/manifest.json", {})
        calibration = read_json(PROJECT_ROOT / "data/acg/review/calibration-report.json", {})
        return {"ready": manifest.get("acg_only") is True and manifest.get("train_count", 0) >= 100000
                    and (PROJECT_ROOT / "data/acg/train.jsonl").is_file() and (PROJECT_ROOT / "data/acg/validation.jsonl").is_file(),
                "train_count": manifest.get("train_count", 0), "labels": manifest.get("train_labels", {}),
                "reference_type": "ACG / 独立模型一致性初标 / 助手抽样复核",
                "train_file": "data/acg/train.jsonl", "validation_file": "data/acg/validation.jsonl",
                "acg_only": manifest.get("acg_only", False), "review": manifest.get("assistant_review", {}),
                "calibration": calibration.get("methods", {}), "caps": manifest.get("caps", {}),
                "content_types": manifest.get("files", {}).get("train", {}).get("content_types", {}),
                "test_reference_count": manifest.get("files", {}).get("test_reference", {}).get("count", 0)}

    @guarded
    def use_expanded_dataset(self):
        info = self._expansion_info()
        if not info["ready"]:
            raise ValueError("扩充训练集尚未构建完成。")
        self._dataset_file = project_path(info["train_file"])
        self._samples = load_samples(self._dataset_file, allow_unlabeled=True)
        return {"dataset": self._dataset_stats(), "preset": info}

    def _dataset_stats(self):
        methods = Counter(row.get("label_method", row.get("source", "未记录")) for row in self._samples)
        return {"count": len(self._samples), "labels": dict(Counter(row["label"] or "未标注" for row in self._samples)),
                "path": str(self._dataset_file), "label_methods": dict(methods),
                "domains": dict(Counter(row.get("domain", "unknown") for row in self._samples)),
                "content_types": dict(Counter(row.get("content_type", "unknown") for row in self._samples)),
                "review_statuses": dict(Counter(row.get("review_status", "not_reviewed") for row in self._samples)),
                "unique_works": len({row["work_id"] for row in self._samples if row.get("work_id")})}

    @guarded
    def hardware(self):
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if result.returncode:
            raise RuntimeError("NVIDIA 驱动未就绪。")
        return {"gpu": result.stdout.strip()}

    def _runs(self):
        runs = []
        for path in sorted((PROJECT_ROOT / "runs").glob("*/best/emotion_model.json"), reverse=True):
            meta = read_json(path, {})
            runs.append({"path": str(path.parent), "name": path.parent.parent.name, "epoch": meta.get("epoch"),
                         "f1": meta.get("validation", {}).get("macro_f1"), "labels": meta.get("labels")})
        return runs

    @guarded
    def list_runs(self):
        return self._runs()

    @guarded
    def start_training(self, values):
        config = TrainConfig.from_dict(values)
        atomic_json(PROJECT_ROOT / "runtime/settings.json", config.to_dict())
        result = self._manager.start_training(config.to_dict())
        return result

    @guarded
    def stop_training(self):
        return self._manager.stop_training()

    @guarded
    def poll(self):
        return {"training": self._manager.training_status(), "download": self._manager.model_download_status(),
                "analysis_running": self._manager.active(self._manager.analysis)}

    @guarded
    def download_model(self):
        return self._manager.prepare_model()

    @guarded
    def load_analysis(self, checkpoint, device):
        return self._manager.start_analysis(checkpoint, device)

    @guarded
    def unload_analysis(self):
        return self._manager.stop_analysis()

    @guarded
    def predict(self, text):
        return self._manager.predict(sentences(text))

    @guarded
    def dataset_page(self, offset=0, limit=25, label="", review_status=""):
        if type(offset) is not int or type(limit) is not int or not 0 <= offset or not 1 <= limit <= 100:
            raise ValueError("分页参数无效。")
        if label not in ["", *LABELS] or review_status not in ["", "not_reviewed", "assistant_reviewed", "user_reviewed"]:
            raise ValueError("标注筛选条件无效。")
        indices = [i for i, row in enumerate(self._samples) if (not label or row["label"] == label)
                   and (not review_status or row.get("review_status", "not_reviewed") == review_status)]
        page = indices[offset:offset + limit]
        return {"rows": [self._samples[i] for i in page], "indices": page, "total": len(indices),
                "offset": offset, "stats": self._dataset_stats()}

    @guarded
    def update_sample(self, index, text, label):
        if type(index) is not int or not 0 <= index < len(self._samples) or label not in ["", *LABELS] or not str(text).strip():
            raise ValueError("样本内容、标签或索引无效。")
        original = self._samples[index]
        if original["text"] == text.strip() and original["label"] == label:
            return self._dataset_stats()
        changed_text = original["text"] != text.strip()
        update = {**original, "text": text.strip(), "label": label,
                  "label_method": "user_edit" if label else "unlabeled", "human_reviewed": bool(label),
                  "review_status": "user_reviewed" if label else "not_reviewed"}
        if "label_method" in original:
            update["previous_label_method"] = original["label_method"]
        if changed_text:
            update["source_text_edited"] = True
            update.setdefault("original_source_text", original["text"])
            update.setdefault("original_domain", original.get("domain", "unknown"))
            update["domain"] = "user_edited"
        if "scores" in original or "confidence" in original:
            update["previous_teacher_annotation"] = {key: original.get(key) for key in ("label", "scores", "confidence", "supporting_teachers")}
            update.pop("scores", None)
            update["confidence"] = None
            update.pop("supporting_teachers", None)
        self._samples[index] = update
        return self._dataset_stats()

    @guarded
    def add_samples(self, text, label=""):
        if label not in ["", *LABELS]:
            raise ValueError("标签无效。")
        added = sentences(text)
        offset = len(self._samples)
        self._samples.extend({"id": f"custom-{offset + i + 1:05}", "text": value, "label": label} for i, value in enumerate(added))
        return self._dataset_stats()

    @guarded
    def import_dataset(self):
        import webview
        paths = self._window.create_file_dialog(webview.FileDialog.OPEN, file_types=("标注数据 (*.jsonl;*.csv;*.json;*.txt)",))
        if not paths:
            return None
        self._samples = load_samples(paths[0], allow_unlabeled=True)
        self._dataset_file = Path(paths[0])
        return self._dataset_stats()

    @guarded
    def save_dataset(self):
        import webview
        path = self._window.create_file_dialog(webview.FileDialog.SAVE, save_filename="custom-train.jsonl",
                                              file_types=("JSONL (*.jsonl)",))
        if not path:
            return None
        target = path if isinstance(path, str) else path[0]
        save_samples(target, self._samples)
        self._dataset_file = Path(target)
        return self._dataset_stats()

    @guarded
    def export_metrics(self, checkpoint=None):
        import webview
        status = self._run_metrics(checkpoint) if checkpoint else self._manager.training_status()
        if not status.get("history"):
            raise ValueError("当前没有可导出的训练指标。")
        path = self._window.create_file_dialog(webview.FileDialog.SAVE, save_filename="training-metrics.json",
                                              file_types=("JSON (*.json)",))
        if path:
            target = path if isinstance(path, str) else path[0]
            atomic_json(target, status)
            return str(target)
        return None

    @guarded
    def load_run_metrics(self, checkpoint):
        return self._run_metrics(checkpoint)

    def _run_metrics(self, checkpoint):
        path = project_path(checkpoint)
        from .model_io import checkpoint_info
        checkpoint_info(path)
        result = read_json(path.parent / "status.json", {})
        result["history"] = read_json(path.parent / "history.json", result.get("history", []))
        return result
