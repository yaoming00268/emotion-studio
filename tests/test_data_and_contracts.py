from collections import Counter
import inspect
import json
from pathlib import Path
import sys
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from emotionlab.common import LABELS, PROJECT_ROOT, normalized_text
from emotionlab.config import TrainConfig
from emotionlab.dataset import audit_datasets, load_samples, save_samples, validate_pair
from emotionlab.inference.client import request
from emotionlab.inference.server import make_server
from emotionlab.metrics import classification_metrics


@pytest.mark.skipif(not (PROJECT_ROOT / "data/manifest.json").is_file(), reason="Local historical corpus is not bundled with the source repository")
def test_exact_corpus_counts_labels_and_no_overlap():
    result = audit_datasets(PROJECT_ROOT / "data")
    assert result["train_count"] == 1000
    assert result["validation_count"] == 140
    assert set(result["train_labels"]) == set(LABELS)
    assert set(result["validation_labels"].values()) == {20}
    stored = json.loads((PROJECT_ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    assert stored["train_sha256"] == result["train_sha256"]
    assert stored["human_reviewed"] is False
    train = load_samples(PROJECT_ROOT / "data/train.jsonl")
    assert len({normalized_text(row["text"]) for row in train}) == 1000
    assert all(len(row["text"]) > 5 for row in train)


def test_duplicate_conflict_and_unlabeled_validation(tmp_path):
    train = [{"text": f"训练的{label}句子", "label": label} for label in LABELS]
    validation = [{"text": f"验证的{label}句子", "label": label} for label in LABELS]
    assert validate_pair(train, validation)["train_count"] == 7
    with pytest.raises(ValueError, match="泄漏"):
        validate_pair(train, train)
    with pytest.raises(ValueError, match="冲突标签"):
        validate_pair(train, [{**row, "label": LABELS[(i+1)%7]} for i, row in enumerate(train)])
    target = tmp_path / "annotation.jsonl"
    save_samples(target, [{"text":"这句话等待人工标注。", "label":""}])
    assert load_samples(target, allow_unlabeled=True)[0]["label"] == ""
    with pytest.raises(ValueError, match="未标注"):
        load_samples(target)


def test_config_rejects_bad_values_and_shared_validation():
    for update in ({"batch_size":True},{"max_length":3},{"learning_rate":float("nan")},
                   {"train_file":"data/train.jsonl","validation_file":"data/train.jsonl"}):
        with pytest.raises(ValueError):
            TrainConfig.from_dict(update)
    default = TrainConfig().validate()
    assert default.memory_budget_mb == 2048
    assert default.batch_size * default.accumulation_steps == 32


def test_metrics_known_confusion_matrix():
    metrics = classification_metrics([0,0,1,1],[0,1,1,1],.5,["喜悦","悲伤"])
    assert metrics["confusion_matrix"] == [[1,1],[0,2]]
    assert metrics["accuracy"] == .75
    assert metrics["per_label"][0]["recall"] == .5
    assert metrics["macro_f1"] == pytest.approx((2/3 + .8)/2)


class FakeAnalyzer:
    def status(self):
        return {"loaded":True,"labels":["喜悦","悲伤"],"device":"cpu"}

    def predict(self, texts, batch_size=8):
        if not isinstance(texts, list) or not texts:
            raise ValueError("句子为空")
        return [{"text":text,"label":"喜悦","confidence":.8,"probabilities":{"喜悦":.8,"悲伤":.2}} for text in texts]


def test_authentication_and_actual_json_inference_transport():
    with make_server("unused",port=0,token="test-token",analyzer=FakeAnalyzer()) as server:
        thread=threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        url=f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with pytest.raises(HTTPError) as denied:
                urlopen(url+"/health",timeout=2)
            assert denied.value.code == 401
            result=request({"url":url,"token":"test-token"},"/predict",{"texts":["真高兴"]})
            assert result["predictions"][0]["label"] == "喜悦"
            with pytest.raises(RuntimeError,match="为空"):
                request({"url":url,"token":"test-token"},"/predict",{"texts":[]})
            with pytest.raises(ValueError,match="本机"):
                request({"url":"https://example.com","token":"x"},"/predict",{})
        finally:
            server.shutdown()
            thread.join(3)


def test_desktop_bridge_keeps_method_signatures_and_privacy(tmp_path):
    from emotionlab.bridge import DesktopAPI
    fixture=tmp_path / "train.jsonl"
    save_samples(fixture,[{"id":f"train-{i+1:04}","text":f"这是第{i+1}个界面测试句子。","label":"中性"} for i in range(10)])
    api=DesktopAPI(dataset_file=str(fixture))
    assert inspect.getfullargspec(DesktopAPI.update_sample).args == ["self","index","text","label"]
    assert api.bootstrap()["data"]["dataset"]["count"] == 10
    assert api.dataset_page(0,10)["data"]["rows"][0]["id"] == "train-0001"
    assert api.update_sample(0,"这是修改后的样本。","")["ok"]
    assert api.dataset_page(0,1)["data"]["rows"][0]["label"] == ""
    assert not api.update_sample(-1,"句子","喜悦")["ok"]
    assert all(name.startswith("_") or callable(getattr(api,name)) for name in vars(api))


def test_desktop_opens_with_empty_editor_when_corpus_is_not_bundled(tmp_path):
    from emotionlab.bridge import DesktopAPI
    api=DesktopAPI(dataset_file=str(tmp_path / "not-bundled.jsonl"))
    try:
        assert api.bootstrap()["data"]["dataset"]["count"] == 0
        assert api.dataset_page()["data"]["rows"] == []
        assert api.add_samples("这是一条等待标注的句子。")["ok"]
        assert api.dataset_page()["data"]["total"] == 1
        assert not (tmp_path / "not-bundled.jsonl").exists()
    finally:
        api._manager.close()


def test_inference_and_desktop_do_not_import_training_or_torch():
    import subprocess
    result=subprocess.run([sys.executable,"-c",
        "import sys; import emotionlab.bridge, emotionlab.inference.client, emotionlab.inference.engine; "
        "assert 'torch' not in sys.modules; assert not any(k.startswith('emotionlab.training') for k in sys.modules)"],
        cwd=PROJECT_ROOT,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_real_service_process_start_stop_and_launcher_ownership(tmp_path):
    from urllib.error import URLError
    from emotionlab.common import atomic_json, read_json
    from emotionlab.processes import ProcessManager

    # /health validates metadata without loading tensor weights or importing torch.
    checkpoint=tmp_path/"health-only-checkpoint"
    checkpoint.mkdir()
    for name in ("adapter_config.json","adapter_model.safetensors","tokenizer_config.json"):
        (checkpoint/name).write_bytes(b"")
    atomic_json(checkpoint/"emotion_model.json",{
        "format":"emotionlab-bert-lora","version":1,"labels":["喜悦","悲伤"],
        "base_model":str(tmp_path/"base"),"max_length":32,"epoch":1})
    manager=ProcessManager(analysis_port=0)
    manager.service_file=tmp_path/"service.json"
    try:
        status=manager.start_analysis(str(checkpoint),"cpu")
        assert status["loaded"] is False
        service=read_json(manager.service_file)
        process=manager.analysis
        assert manager.active(process)
        assert manager.owns_service(process,service)
        manager.stop_analysis()
        assert process.poll() is not None
        assert not manager.service_file.exists()
        with pytest.raises(URLError):
            request(service,"/health",timeout=1)
    finally:
        manager.close()
