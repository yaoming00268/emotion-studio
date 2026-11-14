from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import PROJECT_ROOT, atomic_json, project_path


def main():
    parser = argparse.ArgumentParser(description="BERT 情感工作台：独立 CUDA 训练与本地分析")
    commands = parser.add_subparsers(dest="command", required=True)
    gui = commands.add_parser("gui")
    gui.add_argument("--debug", action="store_true")
    commands.add_parser("diagnose")
    audit = commands.add_parser("audit-data")
    audit.add_argument("--data-dir", default="data/acg", help="目录内应含 train.jsonl 和 validation.jsonl")
    commands.add_parser("download-model")
    train = commands.add_parser("train")
    train.add_argument("--config", type=Path)
    train.add_argument("--run-dir", type=Path, required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("--checkpoint", required=True)
    analyze.add_argument("--text", required=True)
    analyze.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    serve = commands.add_parser("serve")
    serve.add_argument("--checkpoint", required=True)
    serve.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.command == "gui":
        from .gui import main as gui_main
        gui_main(args.debug)
    elif args.command == "diagnose":
        import torch
        result = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "available": torch.cuda.is_available()}
        if result["available"]:
            result.update(gpu=torch.cuda.get_device_name(), total_memory_mb=round(torch.cuda.get_device_properties(0).total_memory / 1024**2))
        print(json.dumps(result, ensure_ascii=False))
        if not result["available"]:
            raise SystemExit(1)
    elif args.command == "audit-data":
        from .dataset import audit_datasets
        print(json.dumps(audit_datasets(project_path(args.data_dir)), ensure_ascii=False, indent=2))
    elif args.command == "download-model":
        from .prepare_model import prepare
        prepare(PROJECT_ROOT / "runtime/model-download.json")
    elif args.command == "train":
        from .config import TrainConfig
        from .training.worker import run_training
        config = TrainConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8"))) if args.config else TrainConfig()
        run_training(config, args.run_dir.resolve())
    elif args.command == "analyze":
        from .common import sentences
        from .inference.engine import BertAnalyzer
        result = BertAnalyzer(project_path(args.checkpoint), args.device).predict(sentences(args.text))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "serve":
        from .inference.server import make_server
        import os
        with make_server(project_path(args.checkpoint), device=args.device) as server:
            target = PROJECT_ROOT / "runtime/inference.json"
            atomic_json(target, {"url": f"http://127.0.0.1:{server.server_address[1]}", "token": server.token,
                                 "pid": os.getpid(), "checkpoint": str(project_path(args.checkpoint))})
            print("本地分析服务已启动：127.0.0.1:18765", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                target.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
