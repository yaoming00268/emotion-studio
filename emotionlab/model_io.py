"""Checkpoint contract only. Training and inference do not import one another."""
from pathlib import Path

from .common import read_json


def checkpoint_info(path):
    path = Path(path).resolve()
    meta = read_json(path / "emotion_model.json")
    if not meta or meta.get("format") != "emotionlab-bert-lora" or meta.get("version") != 1:
        raise ValueError("请选择本项目训练生成的 best 模型目录（需有 emotion_model.json）。")
    labels = meta.get("labels", [])
    if (not isinstance(labels, list) or not 2 <= len(labels) <= 32
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("模型标签集无效。")
    if type(meta.get("max_length")) is not int or not 16 <= meta["max_length"] <= 512:
        raise ValueError("模型序列长度无效。")
    if not isinstance(meta.get("base_model"), str) or not meta["base_model"].strip():
        raise ValueError("模型缺少基础 BERT 目录。")
    for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
        if not (path / name).is_file():
            raise ValueError("分类器文件不完整：" + name)
    return meta


def load_base(path, labels, *, dtype=None):
    from transformers import AutoModelForSequenceClassification
    path = Path(path).resolve()
    if not (path / "config.json").is_file():
        raise ValueError("本地 BERT 模型尚未准备好，请先在界面下载基础模型或选择本地目录。")
    model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True,
        trust_remote_code=False, use_safetensors=True, num_labels=len(labels),
        id2label=dict(enumerate(labels)), label2id={label: i for i, label in enumerate(labels)},
        torch_dtype=dtype, attn_implementation="sdpa")
    if model.config.model_type != "bert":
        raise ValueError("当前项目需要 BERT 架构的中文预训练模型。")
    return model
