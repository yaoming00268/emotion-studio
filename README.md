# Emotion Studio

中文 BERT 七类逐句情感工作台：pywebview / Edge WebView2 GUI、CUDA + LoRA 训练、独立本机分析服务。每句话对应固定参考标签，训练显示交叉熵、准确率、宏 F1、混淆矩阵、各类指标与实际显存占用。

这是纯源码仓库。包含模型训练、分析、GUI、语料采集清洗/自动初标工具、测试和训练配置；不包含模型权重、适配器、任何训练语料/爬取原文/复核记录、运行结果、虚拟环境或下载缓存。

## 安装和启动

Windows、Python 3.11/3.12、NVIDIA CUDA、Edge WebView2 Runtime：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\run.bat
```

无数据时工作台以空标注列表启动，可导入自己的 JSONL/CSV/JSON/TXT，逐句标注并另存。基础模型可在界面点击“准备基础模型”，或运行 `.\.venv\Scripts\python.exe -m emotionlab download-model` 下载固定版本中文 BERT。训练要求 CUDA；分析可选 CUDA 或 CPU。

## 数据与训练

固定标签：中性、喜悦、悲伤、愤怒、恐惧、惊讶、厌恶。JSONL 每行如：

```json
{"id":"example-1","text":"终于再次见到你，真开心！","label":"喜悦"}
```

`configs/acg-bert-cuda.json` 默认读取本地 `data/acg/train.jsonl` 和 `data/acg/validation.jsonl`。需自行准备七类齐全、无重复和跨集合泄漏的标注数据。推荐全部使用可追溯的 ACG 来源，按整作品/系列隔离，并保留来源和自动标签/复核状态。

```powershell
.\.venv\Scripts\python.exe -m emotionlab train --config configs/acg-bert-cuda.json --run-dir runs/my-acg-run
```

默认 LoRA rank 8、FP16、梯度检查点、批大小 4、梯度累积 8、序列长度 256；温和类别抽样采用平方根逆频率，单句权重最多 10 倍。2048 MiB 是 PyTorch 分配器预算，不含 CUDA 上下文和其他进程，也不代表必须占满 2 GiB。原机器的单步验收不代表新机器的显存表现。

`emotionlab/corpus/` 和 `tools/` 保留 ACG 白名单采集、清洗、精确/数字变体及近似词法去重、三个独立教师投票、抽样复核和数据审计代码。采集需要遵循各来源条款；公开评论未获得统一的再次分发许可，因此不上传本地缓存。发布脚本要求真实的校准与逐句复核决定，源码仓库不伪造或附带这些记录；运行采集脚本不会自动得到完整可发布训练集。

## 本地分析与 QQBot

选择 `runs/.../best` 加载分析后，服务监听 `127.0.0.1:18765`，随机令牌与地址写入 `runtime/inference.json`。启动训练会先释放本工作台的分析进程。LoRA 检查点需要相应的基础 BERT 目录。

[QQBot](https://github.com/yaoming00268/qqbot) 仅通过本机 HTTP 接口调用，设置 `emotion.provider: bert` 和相应的 `bert_service_file`。若本仓库克隆在 QQBot 下的 `emotion_studio/`，默认路径即可使用。

```powershell
.\.venv\Scripts\python.exe -m emotionlab analyze --checkpoint runs/my-acg-run/best --text '今天真开心。'
.\.venv\Scripts\python.exe -m pytest --confcutdir=. tests
```

测试使用临时样本与微型本地模型；历史完整语料审计在未提供历史数据时跳过。训练/分析逻辑分别位于 `emotionlab/training/` 和 `emotionlab/inference/`，桌面桥接不导入训练模型。
