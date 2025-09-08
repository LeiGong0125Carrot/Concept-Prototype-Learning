"""
评估和预测工具函数
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


def evaluate_dataset(model, tokenizer, dataset, batch_size=32, device=None, return_predictions=False):
    """
    独立的数据集评估函数
    用于在训练后对特定数据集进行完整评估
    
    Args:
        model: 训练好的模型
        tokenizer: 分词器
        dataset: 要评估的数据集
        batch_size: 批次大小（默认增加到32以加速）
        device: 计算设备
        return_predictions: 是否返回每个样本的预测结果
    
    Returns:
        dict: 包含各种评估指标的字典，如果return_predictions=True，还包含每个样本的预测
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"评估数据集，共 {len(dataset)} 个样本，批次大小: {batch_size}")
    
    # 创建数据加载器 - 优化参数
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, 
        padding=True,
        pad_to_multiple_of=8  # 优化GPU效率
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=data_collator,
        num_workers=4,  # 使用多线程加速数据加载
        pin_memory=True if torch.cuda.is_available() else False  # GPU内存优化
    )
    
    # 将模型移到设备并设置为评估模式
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    # 预测（添加进度条）
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="评估进度", unit="batch")
        for batch in pbar:
            # 移动数据到设备
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(**inputs)
            logits = outputs.logits
            
            # 计算概率
            probs = torch.softmax(logits, dim=-1)
            
            # 收集预测结果
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # 转换为float32以避免BFloat16转换问题
            all_probs.extend(probs.float().cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'batch_acc': (predictions == labels).float().mean().item()
            })
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算详细指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary'
    )
    
    # 每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None
    )
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
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
    加载训练过程中保存的最佳模型检查点
    
    注意：
    - 当 TrainingArguments 中设置 load_best_model_at_end=True 时，
      Trainer 会自动将最佳模型保存到 output_dir 根目录
    - checkpoint-* 目录是训练过程中的中间保存点，不一定是最佳模型
    - 优先加载根目录的模型（Trainer保存的最佳模型）
    
    Args:
        output_dir: 训练输出目录
        model_name: 原始模型名称（用作后备）
        cache_dir: 模型缓存目录
    
    Returns:
        model: 加载的模型
        tokenizer: 分词器
    """
    from pathlib import Path
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import json
    
    checkpoint_dir = Path(output_dir)
    
    # 优先策略1: 尝试从output_dir根目录加载（Trainer保存的最佳模型）
    if (checkpoint_dir / "pytorch_model.bin").exists() or \
       (checkpoint_dir / "model.safetensors").exists():
        logger.info(f"从输出目录加载最佳模型: {checkpoint_dir}")
        model_path = str(checkpoint_dir)
    
    # 策略2: 查找专门的best_model目录（如果使用自定义保存）
    elif (checkpoint_dir / "best_model").exists():
        best_model_dir = checkpoint_dir / "best_model"
        if (best_model_dir / "pytorch_model.bin").exists() or \
           (best_model_dir / "model.safetensors").exists():
            logger.info(f"从best_model目录加载: {best_model_dir}")
            model_path = str(best_model_dir)
        else:
            logger.warning("best_model目录存在但没有找到模型文件")
            model_path = model_name
    
    # 策略3: 根据trainer_state.json找到最佳checkpoint
    elif (checkpoint_dir / "trainer_state.json").exists():
        with open(checkpoint_dir / "trainer_state.json", 'r') as f:
            trainer_state = json.load(f)
        
        best_model_checkpoint = trainer_state.get('best_model_checkpoint')
        if best_model_checkpoint:
            best_checkpoint_path = Path(best_model_checkpoint)
            if best_checkpoint_path.exists():
                logger.info(f"根据trainer_state.json加载最佳检查点: {best_checkpoint_path}")
                model_path = str(best_checkpoint_path)
            else:
                # 尝试相对路径
                best_checkpoint_path = checkpoint_dir / best_checkpoint_path.name
                if best_checkpoint_path.exists():
                    logger.info(f"加载最佳检查点: {best_checkpoint_path}")
                    model_path = str(best_checkpoint_path)
                else:
                    logger.warning(f"trainer_state.json中记录的最佳检查点不存在: {best_model_checkpoint}")
                    model_path = model_name
        else:
            logger.warning("trainer_state.json中没有best_model_checkpoint信息")
            model_path = model_name
    
    # 策略4: 查找checkpoint目录（后备方案）
    else:
        checkpoint_dirs = list(checkpoint_dir.glob("checkpoint-*"))
        if checkpoint_dirs:
            # 找到最新的checkpoint（注意：这不一定是最佳的）
            latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.name.split("-")[-1]))
            logger.warning(f"未找到最佳模型记录，使用最新checkpoint: {latest_checkpoint}")
            model_path = str(latest_checkpoint)
        else:
            logger.warning(f"未找到任何检查点，使用原始模型: {model_name}")
            model_path = model_name
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if model_path != model_name else model_name,
        cache_dir=cache_dir
    )
    
    # 加载模型
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    
    # 检查是否在分布式环境中
    import os
    load_kwargs = {
        "torch_dtype": dtype
    }
    
    # 只有在非分布式环境下才使用device_map
    if not os.environ.get('LOCAL_RANK') and torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        **load_kwargs
    )
    
    logger.info(f"成功加载模型，参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer


def predict_batch(model, tokenizer, texts, batch_size=8, device=None):
    """
    对一批文本进行预测
    
    Args:
        model: 训练好的模型
        tokenizer: 分词器
        texts: 要预测的文本列表
        batch_size: 批次大小
        device: 计算设备
    
    Returns:
        predictions: 预测类别
        probabilities: 各类别概率
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"对 {len(texts)} 条文本进行预测...")
    
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    
    # 分批处理（添加进度条）
    num_batches = (len(texts) + batch_size - 1) // batch_size
    with tqdm(total=num_batches, desc="预测进度", unit="batch") as pbar:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # 移动到设备
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 预测
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(preds.cpu().numpy())
                # 转换为float32以避免BFloat16转换问题
                all_probabilities.extend(probs.float().cpu().numpy())
                
            # 更新进度条
            pbar.update(1)
            pbar.set_postfix({
                'processed': f"{min(i+batch_size, len(texts))}/{len(texts)}"
            })
    
    return np.array(all_predictions), np.array(all_probabilities)