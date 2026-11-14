"""Use a tiny local BERT fixture, never a network or downloaded model."""
import importlib.util

import pytest


@pytest.mark.skipif(not importlib.util.find_spec("torch"), reason="CUDA environment not installed")
def test_tiny_bert_lora_checkpoint_cpu_inference(tmp_path):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    from emotionlab.common import atomic_json
    from emotionlab.inference.engine import BertAnalyzer

    base, checkpoint = tmp_path / "base", tmp_path / "best"
    base.mkdir()
    vocabulary=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]","今","天","真","开","心","难","过","。"]
    (base/"vocab.txt").write_text("\n".join(vocabulary),encoding="utf-8")
    tokenizer=BertTokenizerFast(vocab_file=str(base/"vocab.txt"))
    tokenizer.save_pretrained(base)
    config=BertConfig(vocab_size=len(vocabulary),hidden_size=32,num_hidden_layers=2,num_attention_heads=2,
                      intermediate_size=64,num_labels=2,max_position_embeddings=64)
    model=BertForSequenceClassification(config)
    model.save_pretrained(base,safe_serialization=True)
    model=get_peft_model(model,LoraConfig(task_type=TaskType.SEQ_CLS,r=2,lora_alpha=4,
                                        target_modules=["query","value"],modules_to_save=["classifier"]))
    model.save_pretrained(checkpoint,safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    atomic_json(checkpoint/"emotion_model.json",{"format":"emotionlab-bert-lora","version":1,
        "base_model":str(base),"labels":["喜悦","悲伤"],"max_length":32,"epoch":1})
    analyzer=BertAnalyzer(checkpoint,"cpu")
    assert not analyzer.status()["loaded"]
    first=analyzer.predict(["今天真开心。","今天难过。"])
    second=analyzer.predict(["今天真开心。"])
    assert analyzer.status()["loaded"]
    assert first[0]["probabilities"] == pytest.approx(second[0]["probabilities"], abs=1e-6)
    assert len(first)==2
    assert sum(first[0]["probabilities"].values())==pytest.approx(1)
    assert all(not p.requires_grad for p in analyzer.model.parameters())
