from __future__ import annotations

from pathlib import Path
import threading

from ..common import sentences
from ..model_io import checkpoint_info


class BertAnalyzer:
    def __init__(self, checkpoint, device="cuda"):
        if device not in ("cuda", "cpu"):
            raise ValueError("分析设备请选择 cuda 或 cpu。")
        self.checkpoint, self.device = Path(checkpoint).resolve(), device
        self.meta = checkpoint_info(self.checkpoint)
        self.model = self.tokenizer = None
        self.lock = threading.RLock()

    def load(self):
        import torch
        from transformers import AutoTokenizer
        from peft import PeftModel
        from ..model_io import load_base
        if self.model is not None:
            return
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用；请安装 CUDA 版 PyTorch，或明确选择 CPU 分析。")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        base = load_base(self.meta["base_model"], self.meta["labels"], dtype=dtype)
        model = PeftModel.from_pretrained(base, str(self.checkpoint), local_files_only=True, is_trainable=False)
        model.to(self.device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint), local_files_only=True, trust_remote_code=False)
        self.model, self.tokenizer = model, tokenizer

    def predict(self, texts, batch_size=8):
        import torch
        if (not isinstance(texts, list) or not 1 <= len(texts) <= 1000
                or any(not isinstance(text, str) or not text.strip() or len(text) > 10000 for text in texts)):
            raise ValueError("每次分析需包含 1–1000 句，每句不超过 10000 字。")
        if type(batch_size) is not int or not 1 <= batch_size <= 32:
            raise ValueError("分析批大小必须在 1–32 之间。")
        with self.lock:
            self.load()
            results = []
            labels = self.meta["labels"]
            maximum = self.meta["max_length"]
            with torch.inference_mode():
                for start in range(0, len(texts), batch_size):
                    chunk = texts[start:start + batch_size]
                    lengths = self.tokenizer(chunk, truncation=False, return_length=True)["length"]
                    inputs = self.tokenizer(chunk, padding=True, truncation=True, max_length=maximum, return_tensors="pt")
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    if self.device == "cuda":
                        with torch.autocast("cuda", dtype=torch.float16):
                            logits = self.model(**inputs).logits
                    else:
                        logits = self.model(**inputs).logits
                    probabilities = logits.float().softmax(-1).cpu().tolist()
                    for text, row, length in zip(chunk, probabilities, lengths):
                        index = max(range(len(row)), key=row.__getitem__)
                        results.append({"text": text, "label": labels[index], "confidence": row[index],
                                        "probabilities": dict(zip(labels, row)), "truncated": length > maximum})
            return results

    def status(self):
        return {"checkpoint": str(self.checkpoint), "device": self.device, "loaded": self.model is not None,
                "labels": self.meta["labels"], "max_length": self.meta["max_length"], "epoch": self.meta["epoch"]}
