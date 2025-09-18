#!/usr/bin/env python3
"""
简单测试模型加载
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

def simple_test():
    """简单测试模型"""
    
    print("=== 简单模型测试 ===\n")
    
    # 最新的模型目录
    model_dir = "outputs/Clinical_ModernBERT_20250914_151021_deepspeed_lora"
    
    # 1. 加载并检查配置
    print("1. 检查配置:")
    with open(f"{model_dir}/adapter_config.json") as f:
        config = json.load(f)
    print(f"   inference_mode: {config.get('inference_mode')}")
    print(f"   modules_to_save: {config.get('modules_to_save')}")
    
    # 2. 标准加载方式
    print("\n2. 标准加载模型:")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    peft_config = PeftConfig.from_pretrained(model_dir)
    
    # 加载基础模型
    base_model = AutoModelForSequenceClassification.from_pretrained(
        peft_config.base_model_name_or_path,
        num_labels=2,
        torch_dtype=torch.float32
    )
    
    # 记录基础模型的分类器权重
    base_classifier_weight = base_model.classifier.weight.data.clone()
    base_classifier_bias = base_model.classifier.bias.data.clone()
    
    print(f"   基础模型分类器权重:")
    print(f"     weight mean: {base_classifier_weight.mean():.6f}")
    print(f"     bias: {base_classifier_bias}")
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(
        base_model,
        model_dir,
        is_trainable=False,
        inference_mode=False  # 关键
    )
    model.eval()
    
    # 获取加载后的分类器权重
    lora_classifier_weight = model.base_model.model.classifier.weight.data
    lora_classifier_bias = model.base_model.model.classifier.bias.data
    
    print(f"\n   LoRA模型分类器权重:")
    print(f"     weight mean: {lora_classifier_weight.mean():.6f}")
    print(f"     bias: {lora_classifier_bias}")
    
    # 检查权重是否改变
    weight_diff = torch.mean(torch.abs(base_classifier_weight - lora_classifier_weight)).item()
    bias_diff = torch.mean(torch.abs(base_classifier_bias - lora_classifier_bias)).item()
    
    print(f"\n   权重差异:")
    print(f"     weight diff: {weight_diff:.6f}")
    print(f"     bias diff: {bias_diff:.6f}")
    
    if weight_diff > 0.001 or bias_diff > 0.001:
        print("     ✅ 分类器权重已更新（LoRA训练有效）")
    else:
        print("     ❌ 分类器权重未更新（可能未正确保存）")
    
    # 3. 测试推理
    print("\n3. 测试推理:")
    test_samples = [
        "The patient shows improvement.",
        "Critical condition observed.",
        "Stable vital signs.",
        "Severe adverse reaction noted."
    ]
    
    with torch.no_grad():
        for text in test_samples:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1).item()
            
            print(f"   '{text[:30]}...'")
            print(f"     logits: {logits[0].tolist()}")
            print(f"     probs: [class0={probs[0][0]:.3f}, class1={probs[0][1]:.3f}]")
            print(f"     prediction: {pred}")
    
    # 4. 分析logits模式
    print("\n4. 分析:")
    print("   如果所有样本的logits都相似，说明模型没有学到有效模式")
    print("   如果所有预测都是同一类，说明分类器可能有问题")
    
    return model

if __name__ == "__main__":
    model = simple_test()
    print("\n=== 测试完成 ===")