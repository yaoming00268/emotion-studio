"""Process orchestration for the GUI. This module intentionally cannot load models."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from .common import PROJECT_ROOT, atomic_json, now, project_path, read_json
from .config import TrainConfig
from .inference.client import request
from .model_io import checkpoint_info


class ProcessManager:
    def __init__(self, *, analysis_port=18765):
        self.lock = threading.RLock()
        self.training = self.analysis = self.download = None
        self.run_dir = None
        self.download_dir = None
        self.service_file = PROJECT_ROOT / "runtime/inference.json"
        self.analysis_port = analysis_port
        self.logs = {}

    def spawn(self, name, module, arguments, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("a", encoding="utf-8")
        instance_id = uuid.uuid4().hex
        env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "TOKENIZERS_PARALLELISM": "false",
               "HF_HUB_DISABLE_TELEMETRY": "1", "EMOTIONLAB_INSTANCE_ID": instance_id}
        env.pop("PYTHONPATH", None)
        try:
            process = subprocess.Popen([sys.executable, "-m", module, *arguments], cwd=PROJECT_ROOT,
                                       env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except BaseException:
            stream.close()
            raise
        previous = self.logs.get(name)
        if previous is not None:
            previous.close()
        self.logs[name] = stream
        process._emotionlab_instance_id = instance_id
        return process

    @staticmethod
    def active(process):
        return process is not None and process.poll() is None

    @staticmethod
    def owns_service(process, info):
        # Windows venv python.exe is a launcher: Popen.pid differs from os.getpid().
        expected = getattr(process, "_emotionlab_instance_id", None)
        return bool(expected and info.get("instance_id") == expected)

    @staticmethod
    def terminate_tree(process):
        if not ProcessManager.active(process):
            return
        if os.name == "nt":
            result = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode and process.poll() is None:
                raise RuntimeError("无法退出本工作台启动的进程树。")
        else:
            process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def start_training(self, values):
        with self.lock:
            if self.active(self.training):
                raise ValueError("当前训练尚未结束。")
            if self.active(self.download):
                raise ValueError("请等待基础模型下载完成。")
            config = TrainConfig.from_dict(values)
            from .dataset import load_samples, validate_pair
            validate_pair(load_samples(project_path(config.train_file), config.labels),
                          load_samples(project_path(config.validation_file), config.labels), config.labels)
            if not all((project_path(config.base_model) / name).is_file() for name in ("config.json", "model.safetensors")):
                raise ValueError("请先下载基础 BERT 模型或设置已有的本地模型目录。")
            self.stop_analysis()
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.run_dir = PROJECT_ROOT / "runs" / (stamp + "-" + uuid.uuid4().hex[:6])
            self.run_dir.mkdir(parents=True)
            atomic_json(self.run_dir / "config.json", config.to_dict())
            atomic_json(self.run_dir / "status.json", {"state": "starting", "started_at": now(), "history": [], "config": config.to_dict()})
            self.training = self.spawn("training", "emotionlab.training.worker", ["--config", str(self.run_dir / "config.json"),
                "--run-dir", str(self.run_dir)], self.run_dir / "training.log")
            return {"run_dir": str(self.run_dir), "pid": self.training.pid}

    def stop_training(self):
        with self.lock:
            if self.active(self.training) and self.run_dir:
                (self.run_dir / "STOP").touch()
                return {"state": "stopping"}
            return {"state": "idle"}

    def training_status(self):
        with self.lock:
            if not self.run_dir:
                return {"state": "idle", "history": []}
            state = read_json(self.run_dir / "status.json", {})
            if self.training is not None and self.training.poll() is not None and state.get("state") not in ("completed", "failed", "cancelled"):
                state.update(state="failed", error="训练进程意外退出，详情见 training.log。", exit_code=self.training.returncode)
            return {**state, "run_dir": str(self.run_dir), "running": self.active(self.training)}

    def stop_analysis(self):
        with self.lock:
            self.terminate_tree(self.analysis)
            if self.analysis is not None:
                info = read_json(self.service_file, {})
                if self.owns_service(self.analysis, info):
                    self.service_file.unlink(missing_ok=True)
            self.analysis = None
            return {"state": "unloaded"}

    def start_analysis(self, checkpoint, device="cuda"):
        with self.lock:
            if self.active(self.training):
                raise ValueError("请等待训练结束，或先停止训练，再加载分析模型。")
            if device not in ("cuda", "cpu"):
                raise ValueError("分析设备无效。")
            checkpoint = project_path(checkpoint)
            checkpoint_info(checkpoint)
            if self.active(self.analysis):
                info = read_json(self.service_file, {})
                if (self.owns_service(self.analysis, info)
                        and info.get("checkpoint") == str(checkpoint) and info.get("device") == device):
                    return request(info, "/health", timeout=3)
                self.stop_analysis()
            self.analysis = self.spawn("analysis", "emotionlab.inference.server", ["--checkpoint", str(checkpoint),
                "--device", device, "--port", str(self.analysis_port), "--service-file", str(self.service_file)],
                PROJECT_ROOT / "runtime/inference.log")
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if not self.active(self.analysis):
                        raise RuntimeError("分析服务启动失败，检查 runtime/inference.log（端口 18765 可能被占用）。")
                    info = read_json(self.service_file, {})
                    if self.owns_service(self.analysis, info):
                        try:
                            return request(info, "/health", timeout=1)
                        except OSError:
                            pass
                    time.sleep(.1)
                raise RuntimeError("分析服务未及时启动，请检查日志。")
            except BaseException:
                self.stop_analysis()
                raise

    def predict(self, texts):
        if not self.active(self.analysis):
            raise ValueError("请先选择训练产物并加载分析模型。")
        info = read_json(self.service_file, {})
        if not self.owns_service(self.analysis, info):
            raise ValueError("分析服务描述已变更，请重新加载分析模型。")
        return request(info, "/predict", {"texts": texts})

    def prepare_model(self):
        with self.lock:
            if self.active(self.training) or self.active(self.download):
                raise ValueError("请先等待当前训练或下载任务完成。")
            self.download_dir = PROJECT_ROOT / "runtime/model-download.json"
            atomic_json(self.download_dir, {"state": "starting", "message": "正在连接模型仓库…"})
            self.download = self.spawn("download", "emotionlab.prepare_model", ["--status", str(self.download_dir)],
                                       PROJECT_ROOT / "runtime/model-download.log")
            return {"state": "starting"}

    def model_download_status(self):
        state = read_json(self.download_dir, {}) if self.download_dir else {"state": "idle"}
        if self.download is not None and self.download.poll() is not None and state.get("state") not in ("completed", "failed"):
            state.update(state="failed", error="模型下载进程退出，检查 runtime/model-download.log。")
        return state

    def close(self):
        self.stop_training()
        if self.active(self.training):
            try:
                self.training.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.terminate_tree(self.training)
        self.stop_analysis()
        self.terminate_tree(self.download)
        for stream in self.logs.values():
            stream.close()
