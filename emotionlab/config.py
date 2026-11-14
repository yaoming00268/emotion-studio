from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

from .common import LABELS, project_path


@dataclass
class TrainConfig:
    base_model: str = "models/bert-base-chinese"
    train_file: str = "data/acg/train.jsonl"
    validation_file: str = "data/acg/validation.jsonl"
    labels: list[str] = field(default_factory=lambda: list(LABELS))
    epochs: int = 5
    batch_size: int = 4
    accumulation_steps: int = 8
    max_length: int = 256
    learning_rate: float = 0.0002
    weight_decay: float = 0.01
    lora_rank: int = 8
    lora_alpha: int = 16
    seed: int = 42
    memory_budget_mb: int = 2048
    warmup_ratio: float = 0.1
    resume_from: str = ""
    balance_classes: bool = True
    sampling_power: float = 0.5
    sampling_max_ratio: float = 10.0

    def validate(self):
        if type(self.balance_classes) is not bool:
            raise ValueError("balance_classes 必须为布尔值。")
        integers = {"epochs": (1, 100), "batch_size": (1, 32), "accumulation_steps": (1, 128),
                    "max_length": (16, 512), "lora_rank": (1, 64), "lora_alpha": (1, 128),
                    "seed": (0, 2147483647), "memory_budget_mb": (768, 32768)}
        for key, (minimum, maximum) in integers.items():
            value = getattr(self, key)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{key} 必须是 {minimum}–{maximum} 之间的整数。")
        for key, lo, hi in (("learning_rate", 1e-7, .01), ("weight_decay", 0, 1), ("warmup_ratio", 0, .5),
                           ("sampling_power", 0, 1), ("sampling_max_ratio", 0, 1000)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{key} 必须在 {lo}–{hi} 之间。")
        if (not isinstance(self.labels, list) or not 2 <= len(self.labels) <= 32
                or any(not isinstance(label, str) or not label.strip() for label in self.labels)
                or len(set(self.labels)) != len(self.labels)):
            raise ValueError("标签集必须包含 2–32 个不同的非空标签。")
        for name in ("base_model", "train_file", "validation_file"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} 不能为空。")
        if project_path(self.train_file) == project_path(self.validation_file):
            raise ValueError("训练集和验证集必须使用不同文件。")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("训练配置必须为对象。")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("未知训练设置：" + ", ".join(sorted(unknown)))
        return cls(**values).validate()
