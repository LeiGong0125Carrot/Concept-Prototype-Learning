#!/usr/bin/env python
"""测试不同的tokenizer加载方法"""

from transformers import AutoTokenizer
import sys

model_name = "Simonlee711/Clinical_ModernBERT"
cache_dir = "./models/clinical_modern_bert"

print("=" * 60)
print("测试Clinical_ModernBERT tokenizer加载")
print("=" * 60)

# 方法1: 标准加载
print("\n1. 尝试标准加载...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    print("✓ 标准加载成功!")
    sys.exit(0)
except Exception as e:
    print(f"✗ 标准加载失败: {e}")

# 方法2: use_fast=False
print("\n2. 尝试use_fast=False...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        use_fast=False
    )
    print("✓ use_fast=False成功!")
    sys.exit(0)
except Exception as e:
    print(f"✗ use_fast=False失败: {e}")

# 方法3: trust_remote_code=True
print("\n3. 尝试trust_remote_code=True...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )
    print("✓ trust_remote_code=True成功!")
    sys.exit(0)
except Exception as e:
    print(f"✗ trust_remote_code=True失败: {e}")

# 方法4: 使用基础模型tokenizer
print("\n4. 尝试使用基础ModernBERT tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        "answerdotai/ModernBERT-base",
        cache_dir=cache_dir
    )
    print("✓ 基础ModernBERT tokenizer加载成功!")
    print("\n建议: 修改train_binary_classification.py，使用基础模型的tokenizer")
    print("tokenizer = AutoTokenizer.from_pretrained('answerdotai/ModernBERT-base')")
    sys.exit(0)
except Exception as e:
    print(f"✗ 基础tokenizer也失败: {e}")

print("\n所有方法都失败了，需要进一步调查...")
sys.exit(1)