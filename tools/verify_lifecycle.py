"""Verify real CUDA reload, QQBot transport, resumed training and cancellation."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    from emotionlab.common import atomic_json
    from emotionlab.config import TrainConfig
    from emotionlab.model_io import checkpoint_info
    from emotionlab.processes import ProcessManager

    checkpoint = Path(args.checkpoint).resolve()
    meta = checkpoint_info(checkpoint)
    manager = ProcessManager()
    report = {"checkpoint": str(checkpoint)}
    try:
        manager.start_analysis(str(checkpoint), "cuda")
        analysis = manager.analysis
        texts = ["今天收到朋友的祝福，我真的很开心！", "会议明天下午两点开始。"]
        result = manager.predict(texts)
        assert result["model"]["loaded"]
        assert result["model"]["device"] == "cuda"
        assert len(result["predictions"]) == len(texts)
        report["predictions"] = result["predictions"]

        adapter_file = ROOT.parent / "emotion_bert_client.py"
        if adapter_file.is_file():
            spec = importlib.util.spec_from_file_location("qqbot_bert_adapter", adapter_file)
            adapter = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(adapter)
            assert adapter.predict(manager.service_file, texts) == result["predictions"]
            report["qqbot_http_adapter"] = True
        assert "torch" not in sys.modules
        report["client_does_not_import_torch"] = True

        config = TrainConfig.from_dict({**meta["config"], "resume_from": str(checkpoint), "epochs": 1})
        manager.start_training(config.to_dict())
        assert analysis.poll() is not None and manager.analysis is None
        assert not manager.service_file.exists()
        report["analysis_exits_before_training"] = True

        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            state = manager.training_status()
            if state.get("step", 0) >= 1:
                break
            if state.get("state") == "failed" or not manager.active(manager.training):
                raise RuntimeError(str(state))
            time.sleep(.1)
        else:
            raise RuntimeError("Resumed CUDA training did not reach its first optimizer step")
        report["resumed_optimizer_step"] = state["step"]
        manager.stop_training()
        manager.training.wait(timeout=30)
        stopped = manager.training_status()
        assert stopped["state"] == "cancelled", stopped
        assert manager.training.returncode == 0
        report.update(cooperative_stop=True, cancelled_run=str(manager.run_dir), ok=True)
    finally:
        manager.close()
    atomic_json(ROOT / "runtime/lifecycle-check.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
