"""Smoke a genuine optimizer step against the full ACG CUDA dataset."""
from __future__ import annotations

from pathlib import Path
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from emotionlab.common import atomic_json
    from emotionlab.config import TrainConfig
    from emotionlab.processes import ProcessManager
    config=TrainConfig(train_file="data/acg/train.jsonl",validation_file="data/acg/validation.jsonl",epochs=1,
                       max_length=256, warmup_ratio=0,balance_classes=True,sampling_power=.5,sampling_max_ratio=10)
    manager=ProcessManager()
    try:
        started=manager.start_training(config.to_dict())
        deadline=time.monotonic()+240
        while time.monotonic()<deadline:
            state=manager.training_status()
            if state.get("step",0)>=1:
                break
            if state.get("state")=="failed" or not manager.active(manager.training):
                raise RuntimeError(json.dumps(state,ensure_ascii=False))
            time.sleep(.2)
        else:
            raise RuntimeError("Large dataset did not reach its first CUDA optimizer step")
        manager.stop_training()
        manager.training.wait(timeout=30)
        final=manager.training_status()
        assert final["state"]=="cancelled"
        assert final["dataset"]["train_count"]>=100000
        assert final["sampling"]=="balanced_with_replacement"
        published=json.loads((ROOT/"data/acg/manifest.json").read_text(encoding="utf-8"))
        loaded=json.loads((Path(started["run_dir"])/"dataset_manifest.json").read_text(encoding="utf-8"))
        assert loaded["train_sha256"]==published["files"]["train"]["semantic_sha256"]
        assert final["peak_allocated_mb"]<=config.memory_budget_mb
        assert final["truncated_sentences"]=={"train":0,"validation":0}
        report={"ok":True,"run_dir":started["run_dir"],"train_count":final["dataset"]["train_count"],
                "optimizer_steps":final["step"],"loss":final["live_train_loss"],
                "peak_allocated_mib":final["peak_allocated_mb"],"peak_reserved_mib":final["peak_reserved_mb"],
                "cooperative_stop":True,
                "sampling":final["sampling"],
                "train_semantic_sha256":loaded["train_sha256"],
                "truncated_sentences":final["truncated_sentences"],
                "scope":"Full ACG file loaded/tokenized; genuine CUDA update then stopped. This is not a full training run or new best model."}
        atomic_json(ROOT/"data/acg/cuda-load-check.json",report)
        print(json.dumps(report,ensure_ascii=False,indent=2))
    finally:
        manager.close()


if __name__=="__main__":
    main()
