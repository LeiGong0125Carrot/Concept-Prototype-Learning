"""
评估和预测工具函数
"""

# 标准库imports
import sys
import torch
import numpy as np
import json
import os
import logging
import time
from pathlib import Path

# PyTorch相关imports
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

# Transformers imports
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    AutoConfig,
    DataCollatorWithPadding
)

# PEFT imports
from peft import PeftModel, PeftConfig, LoraConfig, get_peft_model

# 机器学习库imports
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# 其他工具库imports
from tqdm import tqdm
from safetensors import safe_open

# 本地模块imports
from data import ClinicalBinaryDataset
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    """
    计算评估指标 - 用于Trainer
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # 计算准确率
    accuracy = accuracy_score(labels, predictions)
    
    # 计算二分类指标
    precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    # 计算每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'f1': f1_binary,
        'precision': precision_binary,
        'recall': recall_binary,
    }
    
    # 如果有两个类别，添加每个类别的指标
    if len(precision_per_class) >= 2:
        metrics.update({
            'class_0_precision': precision_per_class[0],
            'class_0_recall': recall_per_class[0],
            'class_0_f1': f1_per_class[0],
            'class_1_precision': precision_per_class[1],
            'class_1_recall': recall_per_class[1],
            'class_1_f1': f1_per_class[1],
        })
    
    return metrics


def evaluate_dataset(model, tokenizer, dataset, batch_size=64, device=None, return_predictions=False, use_mixed_precision=True):
    """
    独立的数据集评估函数（优化版）
    用于在训练后对特定数据集进行完整评估
    
    Args:
        model: 训练好的模型
        tokenizer: 分词器
        dataset: 要评估的数据集
        batch_size: 批次大小（默认增加到64以加速）
        device: 计算设备
        return_predictions: 是否返回每个样本的预测结果
        use_mixed_precision: 是否使用混合精度推理（加速GPU计算）
    
    Returns:
        dict: 包含各种评估指标的字典，如果return_predictions=True，还包含每个样本的预测
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 对于长序列，不要自动增加batch_size，尊重用户传递的参数
    # 如果用户明确传递了小的batch_size，说明是为了避免OOM
    logger.info(f"接收到的batch_size参数: {batch_size}")
    
    logger.info(f"评估数据集，共 {len(dataset)} 个样本，批次大小: {batch_size}")
    
    # 创建数据加载器 - 优化参数
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, 
        padding='longest',  # 只padding到当前batch最长，而不是max_length
        pad_to_multiple_of=8  # 优化GPU效率
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=data_collator,
        num_workers=min(8, os.cpu_count() or 4),  # 动态调整线程数
        pin_memory=True if torch.cuda.is_available() else False,  # GPU内存优化
        prefetch_factor=2 if torch.cuda.is_available() else None  # 预取数据
    )
    
    # 将模型移到设备并设置为评估模式
    model = model.to(device)
    model.eval()
    
    # 启用半精度推理（如果GPU支持）
    if use_mixed_precision and torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
        logger.info("使用混合精度推理加速")
    else:
        autocast = lambda: torch.no_grad()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    # 预测（添加进度条）
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="评估进度", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            # 定期清理GPU缓存防止碎片化
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 移动数据到设备（使用non_blocking=True加速）
            inputs = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device, non_blocking=True)
            
            # 前向传播（可选混合精度）
            if use_mixed_precision and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(**inputs)
                    logits = outputs.logits
            else:
                outputs = model(**inputs)
                logits = outputs.logits
            
            # 计算概率
            probs = torch.softmax(logits, dim=-1)
            
            # 收集预测结果（立即转换为numpy释放GPU内存）
            predictions = torch.argmax(logits, dim=-1)
            
            # 先计算batch accuracy用于进度条显示
            batch_acc = (predictions == labels).float().mean().item()
            
            # 转换为numpy并收集
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # 转换为float32以避免BFloat16转换问题
            all_probs.extend(probs.float().cpu().numpy())
            
            # 删除中间变量释放GPU内存
            del outputs, logits, probs, predictions
            
            # 更新进度条
            pbar.set_postfix({
                'batch_acc': batch_acc
            })
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 最终清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 计算详细指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary'
    )
    
    # 每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None
    )

    metrics = {
        'accuracy': accuracy,
        'f1': f1_binary,
        'precision': precision_binary,
        'recall': recall_binary,
        
        # 类别0的指标
        'class_0_precision': precision_per_class[0],
        'class_0_recall': recall_per_class[0],
        'class_0_f1': f1_per_class[0],
        
        # 类别1的指标  
        'class_1_precision': precision_per_class[1],
        'class_1_recall': recall_per_class[1],
        'class_1_f1': f1_per_class[1],
        'total_samples': len(all_labels)
    }
    
    # 如果需要返回详细预测结果
    if return_predictions:
        predictions_detail = []
        for i in range(len(all_predictions)):
            predictions_detail.append({
                'index': i,
                'true_label': int(all_labels[i]),
                'predicted_label': int(all_predictions[i]),
                'probability_class_0': float(all_probs[i][0]),
                'probability_class_1': float(all_probs[i][1]),
                'correct': bool(all_labels[i] == all_predictions[i])
            })
        metrics['predictions'] = predictions_detail
    
    logger.info("评估完成，指标：")
    for key, value in metrics.items():
        if key not in ['confusion_matrix', 'predictions']:
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

    
    return metrics


def load_best_checkpoint(output_dir, model_name, cache_dir="./models/clinical_modern_bert"):
    """
    加载最佳模型检查点 - 标准模式（非LoRA）
    
    Args:
        output_dir: 训练输出目录
        model_name: 原始模型名称（用作后备）
        cache_dir: 模型缓存目录
    
    Returns:
        model: 加载的模型
        tokenizer: 分词器
    """
    
    checkpoint_dir = Path(output_dir)
    
    # 直接使用 best_model 目录
    best_model_dir = checkpoint_dir / "best_model"
    logger.info(f"从best_model目录加载: {best_model_dir}")
    model_path = str(best_model_dir)
    
    # 设置加载参数
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    
    load_kwargs = {"torch_dtype": dtype}
    # 在多GPU环境下先加载到CPU避免内存冲突
    if os.environ.get('LOCAL_RANK'):
        logger.info("多GPU环境：先加载模型到CPU")
        load_kwargs["device_map"] = "cpu"
    elif torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    
    # 直接使用AutoModel加载（Trainer保存的格式）
    logger.info(f"加载模型: {model_path}")
    
    # 强制加载本地权重文件而不是从远程加载
    load_kwargs["local_files_only"] = True
    load_kwargs["trust_remote_code"] = False
    
    # 检查模型文件是否存在
    has_safetensors = (Path(model_path) / "model.safetensors").exists()
    has_pytorch_bin = (Path(model_path) / "pytorch_model.bin").exists()
    has_config = (Path(model_path) / "config.json").exists()
    
    if not has_config:
        logger.error("缺少config.json文件")
        raise FileNotFoundError(f"模型目录 {model_path} 缺少config.json文件")
    
    if not has_safetensors and not has_pytorch_bin:
        logger.error("缺少模型权重文件 (model.safetensors 或 pytorch_model.bin)")
        raise FileNotFoundError(f"模型目录 {model_path} 缺少模型权重文件")
    
    # 检查是否存在LoRA适配器文件，如果存在则使用手动加载
    adapter_config_path = Path(model_path) / "adapter_config.json"
    use_manual_loading = adapter_config_path.exists()
    
    if use_manual_loading:
        logger.info("检测到LoRA适配器文件，使用手动加载确保权重正确性...")
    
    try:
        if not use_manual_loading:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                **load_kwargs
            )
        else:
            raise Exception("使用手动加载避免LoRA适配器干扰")
    except Exception as e:
        logger.warning(f"直接加载失败: {e}")
        logger.info("尝试手动加载模型权重...")
        
        # 手动加载配置和权重  
        
        config = AutoConfig.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_config(config)
        
        # 直接加载safetensors权重
        safetensors_path = Path(model_path) / "model.safetensors"
        if safetensors_path.exists():
            with safe_open(str(safetensors_path), framework="pt") as f:
                state_dict = {key: f.get_tensor(key) for key in f.keys()}
            model.load_state_dict(state_dict, strict=False)
            logger.info("成功加载safetensors权重")
        
        if torch.cuda.is_available():
            model = model.to(dtype)
            if not os.environ.get('LOCAL_RANK'):
                model = model.cuda()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    
    logger.info(f"模型加载完成，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


def load_best_checkpoint_lora(output_dir, model_name, cache_dir="./models/clinical_modern_bert"):
    """
    加载最佳模型检查点 - LoRA模式
    使用PEFT库加载LoRA适配器或从基础模型初始化
    
    Args:
        output_dir: 训练输出目录（包含LoRA适配器）
        model_name: 基础模型名称
        cache_dir: 模型缓存目录
    
    Returns:
        model: 加载的PEFT模型
        tokenizer: 分词器
    """
    
    checkpoint_dir = Path(output_dir)
    
    # 直接使用 best_model 目录
    best_model_dir = checkpoint_dir / "best_model"
    logger.info(f"从best_model目录加载LoRA模型: {best_model_dir}")
    model_path = str(best_model_dir)
    
    # 设置加载参数
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    
    load_kwargs = {"torch_dtype": dtype}
    # 在多GPU环境下先加载到CPU避免内存冲突
    if os.environ.get('LOCAL_RANK'):
        logger.info("多GPU环境：先加载模型到CPU")
        load_kwargs["device_map"] = "cpu"
    elif torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    
    # 检查是否存在LoRA适配器文件
    adapter_config_path = Path(model_path) / "adapter_config.json"
    has_lora_adapter = adapter_config_path.exists()
    
    if has_lora_adapter:
        # 从已保存的checkpoint加载LoRA适配器
        logger.info(f"从checkpoint加载LoRA适配器: {model_path}")
        
        # 加载PEFT配置
        peft_config = PeftConfig.from_pretrained(model_path)
        
        # 加载基础模型
        logger.info(f"加载基础模型: {peft_config.base_model_name_or_path}")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            cache_dir=cache_dir,
            **load_kwargs
        )
        
        # 加载PEFT模型（LoRA适配器）
        model = PeftModel.from_pretrained(base_model, model_path)
        logger.info("成功加载LoRA适配器")
        
    else:
        # 如果没有保存的适配器，从原始模型初始化新的LoRA
        logger.info(f"未找到LoRA适配器，从原始模型初始化: {model_name}")
        
        # 加载基础模型
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            **load_kwargs
        )
        
        # 初始化新的LoRA配置
        lora_config = LoraConfig(
            r=8,                      # LoRA秩
            lora_alpha=16,            # LoRA缩放参数
            target_modules=["query", "value"],  # 目标模块
            lora_dropout=0.1,         # Dropout率
            bias="none",              # 偏置处理方式
            task_type="SEQ_CLS",      # 任务类型
        )
        
        # 应用LoRA到模型
        model = get_peft_model(base_model, lora_config)
        logger.info("初始化新的LoRA适配器完成")
        
        # 打印可训练参数信息
        model.print_trainable_parameters()
    
    # 调试：打印分类头权重信息
    if hasattr(model, 'classifier'):
        classifier = model.classifier if not hasattr(model, 'base_model') else model.base_model.model.classifier
        if classifier is not None:
            classifier_weight = classifier.weight
            classifier_bias = classifier.bias
            logger.info(f"分类头权重形状: {classifier_weight.shape}")
            logger.info(f"分类头权重范围: [{classifier_weight.min():.6f}, {classifier_weight.max():.6f}]")
            if classifier_bias is not None:
                logger.info(f"分类头偏置: {classifier_bias.data}")
    
    # 加载tokenizer
    # 优先从checkpoint加载，如果不存在则从原始模型加载
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        logger.info(f"从checkpoint加载tokenizer")
    except:
        logger.info(f"从原始模型加载tokenizer: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"LoRA模型加载完成 - 总参数: {total_params:,}, 可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer

def predict_batch(model, tokenizer, texts, batch_size=32, device=None, use_mixed_precision=True):
    """
    对一批文本进行预测（优化版）
    
    Args:
        model: 训练好的模型
        tokenizer: 分词器
        texts: 要预测的文本列表
        batch_size: 批次大小（增加到32）
        device: 计算设备
        use_mixed_precision: 是否使用混合精度推理
    
    Returns:
        predictions: 预测类别
        probabilities: 各类别概率
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 动态调整batch_size
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(device).total_memory
        if gpu_memory > 16 * 1024**3:  # 16GB以上GPU
            batch_size = max(batch_size, 64)
        elif gpu_memory > 8 * 1024**3:  # 8GB以上GPU  
            batch_size = max(batch_size, 32)
    
    logger.info(f"对 {len(texts)} 条文本进行预测，批次大小: {batch_size}")
    
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    
    # 分批处理（添加进度条）
    num_batches = (len(texts) + batch_size - 1) // batch_size
    with tqdm(total=num_batches, desc="预测进度", unit="batch") as pbar:
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize（优化：只padding到batch最长）
                inputs = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding='longest',  # 只padding到当前batch最长
                    max_length=512,
                    return_tensors="pt"
                )
                
                # 移动到设备（使用non_blocking=True）
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}
                
                # 预测（可选混合精度）
                if use_mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = model(**inputs)
                        logits = outputs.logits
                else:
                    outputs = model(**inputs)
                    logits = outputs.logits
                
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                # 立即转换为numpy释放GPU内存
                all_predictions.extend(preds.cpu().numpy())
                # 转换为float32以避免BFloat16转换问题
                all_probabilities.extend(probs.float().cpu().numpy())
                
                # 删除中间变量
                del outputs, logits, probs, preds
                
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                'processed': f"{min(i+batch_size, len(texts))}/{len(texts)}"
            })
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # 最终清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return all_predictions, all_probabilities


def load_best_peft_model_for_eval(checkpoint_path: str, 
                                 model_name: str = "Simonlee711/Clinical_ModernBERT"):
    """
    加载最佳PEFT模型用于评估（修复版本）
    优先使用merged_model，如果不存在则使用PEFT adapter
    """
    
    logger.info(f"从 {checkpoint_path} 加载最佳PEFT模型...")
    
    # 检查是否存在merged_model
    checkpoint_parent = Path(checkpoint_path).parent
    merged_model_path = checkpoint_parent / "merged_model"
    
    if merged_model_path.exists() and (merged_model_path / "model.safetensors").exists():
        logger.info(f"发现merged_model，从 {merged_model_path} 加载完整模型...")
        
        # 从原始模型加载tokenizer（merged_model中没有tokenizer文件）
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir="./models/clinical_modern_bert"
        )
        
        # 从merged_model加载完整模型（包含训练好的分类头）
        model = AutoModelForSequenceClassification.from_pretrained(
            str(merged_model_path),
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        )
        logger.info("✅ 从merged_model加载完整模型成功")
        
    else:
        logger.info("未找到merged_model，使用PEFT adapter加载...")
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir="./models/clinical_modern_bert"
        )
        
        # 加载基础模型
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            cache_dir="./models/clinical_modern_bert",
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
        )
        
        # 加载PEFT adapter
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
        logger.info("✅ PEFT adapter加载成功")
    
    # 移动到单个GPU
    if torch.cuda.is_available():
        model = model.to("cuda:0")
    
    model.eval()
    logger.info("✅ 最佳PEFT模型加载成功")
    return model, tokenizer


def evaluate_dataset_simple(model, tokenizer, dataset, batch_size=32):
    """
    优化的数据集评估函数，支持混合精度和加速
    """
    
    start_time = time.time()
    
    # 优化的DataCollator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
        pad_to_multiple_of=8  # 对混合精度友好
    )
    
    # 优化的DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2
    )
    
    device = next(model.parameters()).device
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    logger.info(f"开始评估，共 {len(dataloader)} 个批次...")
    
    # 使用进度条（禁用在非终端环境下的显示，避免卡顿）
    pbar = tqdm(dataloader, desc="评估中", total=len(dataloader), disable=not sys.stdout.isatty())
    
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            try:
                # 每10个batch打印一次进度
                if i % 10 == 0:
                    logger.info(f"处理批次 {i}/{len(dataloader)}")
                    
                # 非阻塞数据移动
                input_ids = batch['input_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                labels = batch['labels']
                
                # 混合精度前向传播
                if torch.cuda.is_available():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                
                # 获取预测
                predictions = torch.argmax(logits, dim=-1)
                
                # 收集结果
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                
                # 释放内存
                del outputs, logits, predictions
                
                # 每50个batch清理一次内存
                if i % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"批次 {i} 处理失败: {e}")
                raise e
    
    # 计算指标
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    
    # 计算每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # 计算性能统计
    total_time = time.time() - start_time
    speed = len(all_labels) / total_time
    
    results = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'f1_class_0': f1_per_class[0] if len(f1_per_class) > 0 else 0.0,
        'f1_class_1': f1_per_class[1] if len(f1_per_class) > 1 else 0.0,
        'recall_class_0': recall_per_class[0] if len(recall_per_class) > 0 else 0.0,
        'recall_class_1': recall_per_class[1] if len(recall_per_class) > 1 else 0.0,
        'precision_class_0': precision_per_class[0] if len(precision_per_class) > 0 else 0.0,
        'precision_class_1': precision_per_class[1] if len(precision_per_class) > 1 else 0.0,
        'num_samples': len(all_labels),
        'eval_time': total_time,
        'eval_speed': speed
    }
    
    # 清理内存
    logger.info("开始清理GPU内存...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("GPU内存清理完成")
    
    logger.info(f"✅ 快速评估完成 - 用时: {total_time:.2f}s, 速度: {speed:.1f} samples/s")
    logger.info("准备返回评估结果...")
    return results