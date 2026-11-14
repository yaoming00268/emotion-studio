"""CUDA LoRA training in its own process; exits to release the CUDA context."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import traceback

from ..common import atomic_json, now, project_path
from ..config import TrainConfig
from ..dataset import dataset_fingerprint, load_samples, validate_pair
from ..metrics import classification_metrics


class Cancelled(Exception):
    pass


def balanced_sampler(rows, seed, *, power=1.0, max_ratio=0.0):
    import torch
    from torch.utils.data import WeightedRandomSampler
    counts = Counter(row["label"] for row in rows)
    largest = max(counts.values())
    by_label = {label: (largest/count)**power for label, count in counts.items()}
    if max_ratio:
        by_label = {label: min(value, max_ratio) for label, value in by_label.items()}
    weights = torch.tensor([by_label[row["label"]] for row in rows], dtype=torch.double)
    return WeightedRandomSampler(weights, len(rows), replacement=True,
                                 generator=torch.Generator().manual_seed(seed))


def run_training(config: TrainConfig, run_dir: Path):
    import numpy as np
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    from ..model_io import checkpoint_info, load_base

    config.validate()
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {"state": "starting", "started_at": now(), "history": [], "config": config.to_dict(),
             "epoch": 0, "step": 0, "best_epoch": None, "best_validation_f1": None}
    history = state["history"]
    atomic_json(run_dir / "config.json", config.to_dict())

    def emit(**values):
        state.update(values, updated_at=now())
        atomic_json(run_dir / "status.json", state)

    def check():
        if (run_dir / "STOP").exists():
            raise Cancelled()

    def memory():
        return {"allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
                "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
                "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
                "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 1)}

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("未检测到 CUDA。训练必须使用 NVIDIA CUDA PyTorch，不会回退 CPU。")
        torch.cuda.set_device(0)
        total = torch.cuda.get_device_properties(0).total_memory
        budget = config.memory_budget_mb * 1024**2
        if budget > total:
            raise ValueError("显存预算超过设备物理显存。")
        torch.cuda.set_per_process_memory_fraction(budget / total, 0)
        torch.cuda.reset_peak_memory_stats()
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        train_rows = load_samples(project_path(config.train_file), config.labels)
        val_rows = load_samples(project_path(config.validation_file), config.labels)
        stats = validate_pair(train_rows, val_rows, config.labels)
        atomic_json(run_dir / "dataset_manifest.json", {**stats,
            "train_sha256": dataset_fingerprint(train_rows), "validation_sha256": dataset_fingerprint(val_rows)})
        base = project_path(config.base_model)
        emit(state="loading", message="正在加载本地中文 BERT 与 LoRA 参数。", dataset=stats,
             gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda)
        check()
        tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
        model = load_base(base, config.labels, dtype=torch.float16)
        if config.resume_from:
            checkpoint = project_path(config.resume_from)
            meta = checkpoint_info(checkpoint)
            if meta["labels"] != config.labels or Path(meta["base_model"]).resolve() != base:
                raise ValueError("继续训练要求同一基础模型和标签顺序。")
            if any(meta.get("config", {}).get(key) != getattr(config, key) for key in ("lora_rank", "lora_alpha")):
                raise ValueError("继续训练不能更改已有模型的 LoRA rank 或 alpha。")
            model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=True, local_files_only=True)
        else:
            model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=config.lora_rank,
                lora_alpha=config.lora_alpha, lora_dropout=.05, target_modules=["query", "value"],
                modules_to_save=["classifier"], bias="none"))
        # GradScaler needs trainable parameters in FP32, frozen BERT stays FP16.
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.to("cuda")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_parameters = sum(p.numel() for p in model.parameters())
        truncated = {}

        def encode_rows(rows, name):
            # Store dense CPU tensors, not hundreds of thousands of tiny tensor
            # views and dictionaries. Tokenize in bounded chunks for large corpora.
            encoded, truncated[name] = {}, 0
            for start in range(0, len(rows), 512):
                check()
                texts = [row["text"] for row in rows[start:start + 512]]
                lengths = tokenizer(texts, truncation=False, add_special_tokens=True, return_length=True)["length"]
                truncated[name] += sum(length > config.max_length for length in lengths)
                chunk = tokenizer(texts, truncation=True, max_length=config.max_length,
                                  padding="max_length", return_tensors="pt")
                for key, values in chunk.items():
                    if key not in encoded:
                        encoded[key] = torch.empty((len(rows), config.max_length), dtype=values.dtype)
                    encoded[key][start:start + len(texts)] = values
            targets = torch.tensor([config.labels.index(row["label"]) for row in rows], dtype=torch.long)

            class EncodedSamples:
                def __len__(self):
                    return len(targets)

                def __getitem__(self, index):
                    return {**{key: value[index] for key, value in encoded.items()}, "labels": targets[index]}

            return EncodedSamples()

        train_data, val_data = encode_rows(train_rows, "train"), encode_rows(val_rows, "validation")
        sampler = balanced_sampler(train_rows, config.seed, power=config.sampling_power,
                                   max_ratio=config.sampling_max_ratio) if config.balance_classes else None
        train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=sampler is None, sampler=sampler,
                                  generator=torch.Generator().manual_seed(config.seed), num_workers=0, pin_memory=True)
        train_eval_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate, weight_decay=config.weight_decay)
        updates_per_epoch = math.ceil(len(train_loader) / config.accumulation_steps)
        total_updates = updates_per_epoch * config.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, round(total_updates * config.warmup_ratio), total_updates)
        scaler = torch.amp.GradScaler("cuda")
        emit(state="training", message="CUDA FP16 / LoRA 训练中。", total_parameters=total_parameters,
             trainable_parameters=trainable, truncated_sentences=truncated, total_steps=total_updates,
             sampling="balanced_with_replacement" if sampler is not None else "natural_shuffle",
             effective_batch_size=config.batch_size * config.accumulation_steps, **memory())
        best_f1, best_loss, step = -1., float("inf"), 0

        def evaluate(loader):
            model.eval()
            targets, predictions, loss_sum, count = [], [], 0., 0
            with torch.inference_mode():
                for batch in loader:
                    check()
                    batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
                    with torch.autocast("cuda", dtype=torch.float16):
                        output = model(**batch)
                    size = batch["labels"].shape[0]
                    loss_sum += float(output.loss) * size
                    count += size
                    targets.extend(batch["labels"].tolist())
                    predictions.extend(output.logits.argmax(dim=-1).tolist())
            return classification_metrics(targets, predictions, loss_sum / count, config.labels)

        started = time.perf_counter()
        for epoch in range(1, config.epochs + 1):
            check()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            running_loss, seen = 0., 0
            for batch_index, batch in enumerate(train_loader):
                check()
                batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
                size = batch["labels"].shape[0]
                # Weight microbatches by the actual accumulation-group size.
                group_start = batch_index // config.accumulation_steps * config.accumulation_steps
                group_samples = min(config.batch_size * config.accumulation_steps,
                                    len(train_data) - group_start * config.batch_size)
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model(**batch)
                    loss = output.loss * size / group_samples
                if not torch.isfinite(loss):
                    raise RuntimeError("出现非有限损失，请降低学习率。")
                scaler.scale(loss).backward()
                running_loss += float(output.loss.detach()) * size
                seen += size
                if (batch_index + 1) % config.accumulation_steps == 0 or batch_index + 1 == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
                    old_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale() >= old_scale:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    emit(epoch=epoch, step=step, batch=batch_index + 1, batches=len(train_loader),
                         live_train_loss=running_loss / seen, learning_rate=scheduler.get_last_lr()[0], **memory())
            emit(state="evaluating", message=f"正在评估第 {epoch} 轮训练集与独立验证集。")
            train_metrics, val_metrics = evaluate(train_eval_loader), evaluate(val_loader)
            record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics,
                      "elapsed_seconds": round(time.perf_counter() - started, 2), **memory()}
            history.append(record)
            atomic_json(run_dir / "history.json", history)
            if val_metrics["macro_f1"] > best_f1 or (val_metrics["macro_f1"] == best_f1 and val_metrics["loss"] < best_loss):
                best_f1, best_loss = val_metrics["macro_f1"], val_metrics["loss"]
                temporary, best = run_dir / "best.pending", run_dir / "best"
                temporary.mkdir(exist_ok=True)
                model.save_pretrained(str(temporary), safe_serialization=True)
                tokenizer.save_pretrained(str(temporary))
                atomic_json(temporary / "emotion_model.json", {"format": "emotionlab-bert-lora", "version": 1,
                    "base_model": str(base), "labels": config.labels, "max_length": config.max_length,
                    "epoch": epoch, "validation": val_metrics, "config": config.to_dict(), "created_at": now(),
                    "trainable_parameters": trainable, "total_parameters": total_parameters,
                    "dataset_fingerprints": {"train": dataset_fingerprint(train_rows), "validation": dataset_fingerprint(val_rows)}})
                # A prior best stays intact until a complete new directory exists.
                previous = run_dir / "best.previous"
                if previous.exists():
                    shutil.rmtree(previous)
                if best.exists():
                    best.rename(previous)
                try:
                    temporary.rename(best)
                except BaseException:
                    if previous.exists() and not best.exists():
                        previous.rename(best)
                    raise
                if previous.exists():
                    shutil.rmtree(previous)
                state.update(best_epoch=epoch, best_validation_f1=best_f1, best_model=str(best))
            emit(state="training", message=f"第 {epoch} 轮已完成。")
        emit(state="completed", message="训练完成，最佳模型已保存。", finished_at=now(), **memory())
    except Cancelled:
        emit(state="cancelled", message="训练已停止，保留此前完成轮次的最佳模型。", finished_at=now())
    except torch.cuda.OutOfMemoryError:
        emit(state="failed", error="超出 CUDA 显存预算。请将批大小减为 1 或 2，序列长度减为 64。", finished_at=now(), **memory())
        raise
    except BaseException as error:
        emit(state="failed", error=f"{type(error).__name__}: {error}", finished_at=now())
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    config = TrainConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8")))
    run_training(config, Path(args.run_dir).resolve())


if __name__ == "__main__":
    main()
