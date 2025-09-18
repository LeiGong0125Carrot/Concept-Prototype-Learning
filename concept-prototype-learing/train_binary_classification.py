"""
使用AutoModelForSequenceClassification和Trainer进行二分类微调
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import logging
import json
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from transformers import set_seed
# LoRA imports removed
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader, Subset

from data import create_dataloaders, ClinicalBinaryDataset
from utility import evaluate_dataset, load_best_checkpoint, load_best_checkpoint_lora, predict_batch, compute_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_class_weights(train_dataset):
    """
    计算类别权重
    公式: weight_for_class_i = total_samples / (num_samples_in_class_i * num_classes)
    """
    # 统计每个类别的样本数
    label_counts = {}
    for i in range(len(train_dataset)):
        # 处理Subset类型的数据集
        if isinstance(train_dataset, Subset):
            label = train_dataset.dataset[train_dataset.indices[i]]['labels']
        else:
            label = train_dataset[i]['labels']
        if isinstance(label, torch.Tensor):
            label = label.item()
        label_counts[label] = label_counts.get(label, 0) + 1
    
    total_samples = len(train_dataset)
    num_classes = len(label_counts)
    
    # 计算权重
    class_weights = {}
    for class_id, count in label_counts.items():
        weight = total_samples / (count * num_classes)
        class_weights[class_id] = weight
    
    logger.info("类别权重计算:")
    logger.info(f"  总样本数: {total_samples}")
    logger.info(f"  类别数: {num_classes}")
    logger.info(f"  类别分布: {label_counts}")
    for class_id, weight in class_weights.items():
        logger.info(f"  类别 {class_id}: 权重 = {weight:.4f}")
    
    # 转换为tensor，按类别ID排序
    weights_tensor = torch.tensor([class_weights[i] for i in sorted(class_weights.keys())], 
                                dtype=torch.float32)
    
    return weights_tensor



def load_datasets(data_dir: str, tokenizer, max_length: int = 512, train_ratio: float = 1.0):
    """
    加载训练、验证、测试数据集
    
    Args:
        data_dir: 数据目录路径
        tokenizer: 分词器
        max_length: 最大序列长度
        train_ratio: 训练集使用比例 (0.0, 1.0]，用于快速测试
    """
    datasets = {}
    
    # 训练集
    train_path = Path(data_dir) / "train.json"
    if train_path.exists():
        full_train_dataset = ClinicalBinaryDataset(
            json_path=str(train_path),
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        # 根据train_ratio进行采样
        if train_ratio < 1.0:
            random.seed(42)  # 保证可重复性
            total_samples = len(full_train_dataset)
            sample_size = max(1, int(total_samples * train_ratio))  # 至少保留1个样本
            
            # 随机采样索引
            indices = list(range(total_samples))
            random.shuffle(indices)
            sampled_indices = indices[:sample_size]
            
            # 创建子数据集
            datasets['train'] = Subset(full_train_dataset, sampled_indices)
            logger.info(f"训练集采样: {sample_size}/{total_samples} (比例: {train_ratio:.2%})")
        else:
            datasets['train'] = full_train_dataset
            logger.info(f"训练集样本数: {len(full_train_dataset)} (使用全部数据)")
    
    # 验证集
    dev_path = Path(data_dir) / "dev.json"
    if dev_path.exists():
        datasets['dev'] = ClinicalBinaryDataset(
            json_path=str(dev_path),
            tokenizer=tokenizer,
            max_length=max_length
        )
        logger.info(f"验证集样本数: {len(datasets['dev'])}")
    
    # 测试集
    test_path = Path(data_dir) / "test.json"
    if test_path.exists():
        datasets['test'] = ClinicalBinaryDataset(
            json_path=str(test_path),
            tokenizer=tokenizer,
            max_length=max_length
        )
        logger.info(f"测试集样本数: {len(datasets['test'])}")
    
    return datasets


def set_deterministic_training():
    """设置确定性训练环境"""
    import torch
    import os
    import random
    import numpy as np
    
    # 1. 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 2. 统一TF32设置 - 关键！确保训练和评估一致
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # 3. 设置确定性行为
    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 4. 环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    logger.info("✅ 确定性训练环境已配置")
    logger.info(f"  - TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
    logger.info(f"  - TF32 cudnn: {torch.backends.cudnn.allow_tf32}")
    logger.info(f"  - CUDNN deterministic: {torch.backends.cudnn.deterministic}")

def load_model_and_tokenizer(model_name: str, 
                           cache_dir: str = "./models/clinical_modern_bert",
                           num_labels: int = 2,
                           class_weights: torch.Tensor = None):
    """
    加载模型和tokenizer
    
    Args:
        model_name: 模型名称或路径
        cache_dir: 缓存目录
        num_labels: 分类标签数
        class_weights: 类别权重
    """
    # 首先设置确定性环境
    set_deterministic_training()
    
    logger.info("加载模型和tokenizer...")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    
    # 加载分类模型
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    
    # 检查是否使用DeepSpeed - 如果使用则不设置device_map
    load_kwargs = {
        "num_labels": num_labels,
        "cache_dir": cache_dir,
        "torch_dtype": dtype
    }
    
    # 只有在非DeepSpeed情况下才设置device_map
    # DeepSpeed会自己管理设备分配
    if not os.environ.get('LOCAL_RANK') and torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        **load_kwargs
    )
    
    # 如果有class weights，设置到模型中
    if class_weights is not None:
        logger.info(f"设置类别权重: {class_weights}")
        model.class_weights = class_weights.to(model.device) if torch.cuda.is_available() else class_weights
    
    logger.info(f"模型类型: {type(model).__name__}")
    logger.info(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer



class BinaryClassificationTrainer(Trainer):
    """二分类训练器 - 继承Trainer并添加class weights支持"""
    
    def __init__(self, class_weights=None, model_name: str = "Simonlee711/Clinical_ModernBERT", 
                 cache_dir: str = "./models/clinical_modern_bert", 
                 output_dir: str = "./output/binary_classification", *args, **kwargs):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.class_weights = class_weights
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"初始化训练器 - 使用设备: {self.device}")
        if class_weights is not None:
            logger.info(f"启用类别权重: {class_weights}")
        
        # 调用父类初始化
        super().__init__(*args, **kwargs)
        
        # 重新绑定compute_metrics方法（父类初始化可能覆盖了它）  
        self.compute_metrics = self._internal_compute_metrics
        
        # 初始化固定的loss函数，避免每次compute_loss都重新创建
        self._loss_fct = None
        self._setup_loss_function()
    
    def _setup_loss_function(self):
        """设置固定的loss函数"""
        if self.class_weights is not None:
            # 将class_weights移动到正确的设备和数据类型
            # 注意：这里暂时使用float32，后续在第一次使用时会调整
            weights = self.class_weights.to(self.device, dtype=torch.float32)
            self._loss_fct = nn.CrossEntropyLoss(weight=weights)
            logger.info("初始化固定的加权CrossEntropyLoss")
        else:
            self._loss_fct = nn.CrossEntropyLoss()
            logger.info("初始化固定的标准CrossEntropyLoss")
    
    def _internal_compute_metrics(self, eval_pred):
        """计算训练过程中的评估指标"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        
        # 计算整体指标（binary average），设置zero_division=0来避免警告
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        
        # 计算每个类别的F1分数，设置zero_division=0来避免警告
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
        # 添加每个类别的F1分数
        for i, f1_score in enumerate(f1_per_class):
            metrics[f'f1_class_{i}'] = f1_score if not np.isnan(f1_score) else 0.0
        
        for i, recall_score in enumerate(recall_per_class):
            metrics[f'recall_class_{i}'] = recall_score if not np.isnan(recall_score) else 0.0
        
        for i, precision_score in enumerate(precision_per_class):
            metrics[f'precision_class_{i}'] = precision_score if not np.isnan(precision_score) else 0.0
        
        return metrics
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """自定义损失函数支持类别权重"""
        labels = inputs.get("labels")
        
        # 移除labels从inputs中，避免模型自动计算loss导致设备不匹配
        inputs_without_labels = {k: v for k, v in inputs.items() if k != "labels"}
        
        # 前向传播，不让模型计算loss
        outputs = model(**inputs_without_labels)
        logits = outputs.get('logits')
        
        if labels is not None:
            # 获取模型配置 - 处理DistributedDataParallel包装的情况
            if hasattr(model, 'module'):
                num_labels = model.module.config.num_labels
            else:
                num_labels = model.config.num_labels
            
            # 确保设备和数据类型一致性（只在第一次或设备改变时调整）
            device = logits.device
            dtype = logits.dtype
            
            if self.class_weights is not None:
                # 检查是否需要更新loss函数的权重（设备或类型改变时）
                current_weight = self._loss_fct.weight
                if current_weight.device != device or current_weight.dtype != dtype:
                    # 只有在设备或类型改变时才重新创建
                    weights = self.class_weights.to(device=device, dtype=dtype)
                    self._loss_fct = nn.CrossEntropyLoss(weight=weights)
                    logger.debug(f"Updated loss function for device={device}, dtype={dtype}")
            
            # 使用固定的loss函数（避免重复创建）
            loss = self._loss_fct(logits.view(-1, num_labels), labels.view(-1))
        else:
            loss = None
        
        return (loss, outputs) if return_outputs else loss
    
    def create_training_args(self, 
                           num_epochs: int = 3,
                           batch_size: int = 4,
                           learning_rate: float = 2e-5,
                           warmup_ratio: float = 0.1,
                           weight_decay: float = 0.01,
                           report_to: str = "none"):
        """创建训练参数"""
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            
            # 评估和保存策略
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            
            # 日志设置
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=5,
            report_to=report_to,  # 可配置: "none", "wandb", "tensorboard", etc.
            
            # 优化设置
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            dataloader_pin_memory=torch.cuda.is_available(),
            gradient_checkpointing=False,  # Flash Attention下可以关闭
            group_by_length=True,  # 按长度分组批次，进一步减少padding
            
            # 其他设置
            seed=42,
            push_to_hub=False,
            remove_unused_columns=True
        )
        
        return training_args


def save_training_config(output_dir: str, args, class_weights=None):
    """
    保存训练配置，以便后续评估时使用
    """
    config = {
        'model_name': args.model_name,
        'max_length': args.max_length,
        'batch_size': args.batch_size,
        'eval_batch_size': args.batch_size * 2,
        'pad_to_multiple_of': args.pad_to_multiple_of,
        'use_class_weights': args.use_class_weights,
        'class_weights': class_weights.tolist() if class_weights is not None else None,
        'bf16': args.bf16,
        'seed': args.seed,
        'metric_for_best_model': args.metric_for_best_model,
        'greater_is_better': args.greater_is_better,
        # 添加更多关键配置
        'tf32_disabled': True,  # 我们总是禁用TF32
        'group_by_length': not args.no_group_by_length,
    }
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    config_path = Path(output_dir) / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"训练配置已保存至: {config_path}")
    return config


def load_training_config(output_dir: str):
    """
    加载训练配置
    """
    config_path = Path(output_dir) / "training_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"已加载训练配置: {config_path}")
        return config
    else:
        logger.warning(f"训练配置文件不存在: {config_path}")
        return None


def train_clinical_model(
    model_name: str = "Simonlee711/Clinical_ModernBERT",
    data_dir: str = "./data",
    output_dir: str = None,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    pad_to_multiple_of: int = 64,
    report_to: str = "none",
    use_class_weights: bool = False,
    use_early_stopping: bool = True,
    patience: int = 2,
    train_ratio: float = 1.0,
    deepspeed_config: str = None,
    bf16: bool = None,
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    args=None  # 传入完整的args对象用于保存配置
):
    """统一的训练函数"""
    
    # 0. 自动生成带时间戳的输出目录（如果未指定）
    if output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short_name = model_name.split('/')[-1] if '/' in model_name else model_name
        output_dir = f"./outputs/{model_short_name}_{timestamp}"
    
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")
    
    # 1. 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        cache_dir="./models/clinical_modern_bert"
    )
    
    # 2. 加载数据集
    logger.info("加载数据集...")
    datasets = load_datasets(data_dir, tokenizer, max_length, train_ratio)
    
    train_dataset = datasets.get('train')
    if not train_dataset:
        raise ValueError("训练数据集未找到")
    
    dev_dataset = datasets.get('dev')
    
    # 3. 计算类别权重（如果启用）
    class_weights = None
    if use_class_weights:
        logger.info("计算类别权重...")
        class_weights = calculate_class_weights(train_dataset)
    
    # 3.5 保存训练配置（重要！用于后续评估）
    if args:
        save_training_config(output_dir, args, class_weights)
    
    # 4. 创建DataCollator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt"
    )
    logger.info(f"使用DataCollatorWithPadding - pad_to_multiple_of={pad_to_multiple_of}")
    
    # 5. 创建训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        
        # 评估策略
        eval_strategy="epoch" if dev_dataset else "no",
        save_strategy="epoch",
        save_total_limit=2,
        save_on_each_node=False,  # 只在主节点保存，避免DeepSpeed冲突
        load_best_model_at_end=True if dev_dataset else False,
        metric_for_best_model=metric_for_best_model if dev_dataset else None,
        greater_is_better=greater_is_better if dev_dataset else None,
        
        # 日志设置
        logging_dir=f"{output_dir}/logs",
        logging_steps=5,
        report_to=report_to,
        
        # 优化设置
        bf16=bf16 if bf16 is not None else (torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        dataloader_pin_memory=torch.cuda.is_available(),
        gradient_checkpointing=False,
        group_by_length=True,
        
        # DeepSpeed设置
        deepspeed=deepspeed_config,
        
        # 其他设置
        seed=42,
        push_to_hub=False,
        remove_unused_columns=True,
    )
    
    # 6. 创建回调函数
    callbacks = []
    if use_early_stopping and dev_dataset:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    # 7. 创建Trainer
    trainer = BinaryClassificationTrainer(
        class_weights=class_weights,  # 传入class weights
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # 9. 开始训练
    logger.info("开始训练...")
    logger.info(f"训练参数: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
    logger.info(f"类别权重: {'启用' if use_class_weights else '禁用'}")
    
    # 检查优化器类型（调试信息）
    def check_optimizer_type(trainer):
        """检查优化器类型"""
        try:
            logger.info("=== 优化器信息检查 ===")
            
            # 检查 DeepSpeed 引擎的优化器
            if hasattr(trainer.model, 'optimizer'):
                opt_type = type(trainer.model.optimizer).__name__
                opt_module = type(trainer.model.optimizer).__module__
                logger.info(f"DeepSpeed optimizer: {opt_module}.{opt_type}")
                
                # 检查是否包含 "fused" 关键字
                if 'fused' in opt_type.lower() or 'fused' in opt_module.lower():
                    logger.info("✅ 正在使用 Fused 优化器加速")
                else:
                    logger.info("⚠️  使用标准优化器")
            
            # 检查 DeepSpeed 配置中的优化器名称
            if hasattr(trainer.model, 'optimizer_name'):
                opt_name = trainer.model.optimizer_name()
                logger.info(f"DeepSpeed optimizer config: {opt_name}")
                
            logger.info("========================")
                
        except Exception as e:
            logger.warning(f"检查优化器信息时出错: {e}")
    
    train_result = trainer.train()
    
    # 训练开始后检查优化器（DeepSpeed引擎已初始化）
    check_optimizer_type(trainer)


    # 10. 保存模型
    # 当 load_best_model_at_end=True 时，trainer.model 已经是最佳模型
    # 使用 save_pretrained 保存完整模型（提升复现性）
    logger.info(f"保存最佳模型到: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    trainer.save_state()  # 保存训练状态（用于恢复训练）
    
    # 保存tokenizer到同一目录
    if 'tokenizer' in locals():
        tokenizer.save_pretrained(output_dir)
    else:
        trainer.processing_class.save_pretrained(output_dir)
    
    # 保存最佳模型的元信息
    if hasattr(trainer.state, 'best_metric'):
        best_model_info = {
            "best_metric": trainer.state.best_metric,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
            "metric_for_best_model": training_args.metric_for_best_model,
        }
        with open(Path(output_dir) / "best_model_info.json", 'w') as f:
            json.dump(best_model_info, f, indent=2)
    
    # 11. 保存训练结果
    # Trainer已经在训练过程中进行了评估，结果都在train_result.metrics中
    results = {'train': train_result.metrics}
    
    # 保存结果
    results_path = Path(output_dir) / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    logger.info(f"✅ 训练完成！结果保存至: {results_path}")
    
    return trainer, results




def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Clinical ModernBERT 二分类微调")
    
    # 数据相关
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="数据目录路径")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录路径")
    parser.add_argument("--model_name", type=str, default="Simonlee711/Clinical_ModernBERT",
                        help="预训练模型名称或路径")
    
    # 训练超参数
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--max_length", type=int, default=256,
                        help="最大序列长度")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="预热比例")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    
    # 优化设置
    parser.add_argument("--pad_to_multiple_of", type=int, default=64,
                        help="padding到此倍数（Flash Attention优化）")
    parser.add_argument("--no_group_by_length", action="store_true",
                        help="禁用按长度分组批次")
    parser.add_argument("--use_class_weights", action="store_true",
                        help="使用类别权重处理数据不平衡")
    parser.add_argument("--bf16", action="store_true",
                        help="启用bf16混合精度训练")
    
    # 日志和报告
    parser.add_argument("--report_to", type=str, default="none",
                        choices=["none", "wandb", "tensorboard", "all"],
                        help="日志报告工具")
    parser.add_argument("--logging_steps", type=int, default=5,
                        help="日志记录步数")
    
    # 早停设置
    parser.add_argument("--use_early_stopping", action="store_true", default=True,
                        help="是否使用早停")
    parser.add_argument("--patience", type=int, default=2,
                        help="早停耐心值")
    
    # 任务控制
    parser.add_argument("--do_train", action="store_true", default=False,
                        help="是否执行训练")
    parser.add_argument("--do_eval", action="store_true",
                        help="是否执行dev集评估")
    parser.add_argument("--do_predict", action="store_true",
                        help="是否执行test集评估")
    
    # DeepSpeed设置
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="DeepSpeed配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="分布式训练的local rank")
    
    # LoRA相关参数已移除
    
    # 其他
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--train_ratio", type=float, default=1.0,
                        help="训练集使用比例 (0.0, 1.0]，用于快速测试功能")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss",
                        help="用于选择最佳模型的评估指标")
    parser.add_argument("--greater_is_better", action="store_true",
                        help="指标是否越大越好（默认False，适用于loss）")
    
    return parser.parse_args()


def set_deterministic_behavior(seed=42):
    """设置确定性行为，确保可复现性"""
    logger.info(f"设置确定性行为，随机种子: {seed}")
    
    # 使用transformers的set_seed（更全面）
    from transformers import set_seed
    set_seed(seed)
    
    # 额外设置Python随机种子
    random.seed(seed)
    
    # 额外设置NumPy随机种子
    np.random.seed(seed)
    
    # 额外设置PyTorch随机种子
    torch.manual_seed(seed)
    
    # 设置CUDA随机种子（如果可用）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 设置环境变量
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    logger.info("确定性行为设置完成")

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 第一步：设置确定性行为 - 在所有操作之前
    set_deterministic_behavior(args.seed)
    
    # 额外确保transformers库的确定性设置
    set_seed(args.seed)
    
    logger.info("="*60)
    logger.info("Clinical ModernBERT 二分类微调")
    logger.info("="*60)
    logger.info(f"参数配置:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    logger.info("="*60)
    
    trainer = None
    results = {}
    
    # 执行训练
    if args.do_train:
        logger.info("\n开始训练任务...")
        trainer, train_results = train_clinical_model(
            model_name=args.model_name,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            pad_to_multiple_of=args.pad_to_multiple_of,
            report_to=args.report_to,
            use_class_weights=args.use_class_weights,
            use_early_stopping=args.use_early_stopping,
            patience=args.patience,
            train_ratio=args.train_ratio,
            deepspeed_config=args.deepspeed,
            bf16=args.bf16,
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=args.greater_is_better,
            args=args  # 传入完整的args用于保存配置
        )
        results.update(train_results)
        
        # 训练结束后立即使用trainer进行评估
        '''if trainer and trainer.eval_dataset:
            logger.info("\n" + "="*40)
            logger.info("训练结束后立即评估验证集...")
            logger.info("="*40)
            
            # 重要：检查trainer的状态和最佳模型信息
            if hasattr(trainer.state, 'best_metric') and hasattr(trainer.state, 'best_model_checkpoint'):
                logger.info(f"最佳模型信息:")
                logger.info(f"  best_metric: {trainer.state.best_metric}")
                logger.info(f"  best_model_checkpoint: {trainer.state.best_model_checkpoint}")
                
                # 正确判断当前模型是否为最佳模型
                # 当 load_best_model_at_end=True 时，训练结束后模型会自动恢复到最佳检查点
                if trainer.args.load_best_model_at_end:
                    logger.info(f"  当前模型是最佳模型（已通过load_best_model_at_end自动加载）")
                else:
                    # 通过比较最后保存的检查点路径判断
                    last_checkpoint = getattr(trainer.state, 'global_step', None)
                    if last_checkpoint and trainer.state.best_model_checkpoint:
                        is_best = trainer.state.best_model_checkpoint.endswith(f"checkpoint-{last_checkpoint}")
                        logger.info(f"  当前模型是否为最佳: {is_best}")
                    else:
                        logger.info(f"  无法判断当前模型是否为最佳（缺少必要信息）")
            
            # 检查trainer当前使用的class_weights
            logger.info(f"Trainer当前class_weights: {trainer.class_weights}")
            
            # 直接使用训练时的trainer进行评估
            immediate_dev_metrics = trainer.evaluate()
            
            # 保存立即评估的结果
            results['immediate_dev_eval'] = immediate_dev_metrics
            
            # 打印结果
            logger.info("训练后立即评估结果:")
            for key, value in immediate_dev_metrics.items():
                logger.info(f"  {key}: {value:.6f}")
            
            # 与训练过程中的eval_loss进行比较
            if hasattr(trainer.state, 'best_metric'):
                logger.info(f"训练中最佳eval_loss: {trainer.state.best_metric:.6f}")
                logger.info(f"立即评估eval_loss: {immediate_dev_metrics.get('eval_loss', 0):.6f}")
                diff = abs(trainer.state.best_metric - immediate_dev_metrics.get('eval_loss', 0))
                logger.info(f"差异: {diff:.10f}")
                
                if diff < 1e-6:
                    logger.info("✅ eval_loss完全一致!")
                elif diff < 1e-4:
                    logger.info("⚠️  eval_loss基本一致 (差异很小)")
                else:
                    logger.info("❌ eval_loss不一致 (存在明显差异)")
            
            # 保存到文件 - 使用实际的输出目录
            output_dir = trainer.args.output_dir if hasattr(trainer, 'args') else args.output_dir
            if output_dir:
                immediate_results_path = Path(output_dir) / "immediate_dev_eval_results.json"
                with open(immediate_results_path, 'w') as f:
                    json.dump(immediate_dev_metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
                logger.info(f"立即评估结果已保存至: {immediate_results_path}")
    '''
    # 执行评估（仅评估dev dataset）
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if args.do_eval:
        logger.info("\n" + "="*40)
        logger.info("执行评估任务 (Dev Dataset) - 使用trainer加载最佳模型")
        logger.info(f"当前进程 LOCAL_RANK: {local_rank}")
        logger.info("="*40)
        
        # 检查模型目录是否存在
        output_dir = trainer.args.output_dir if trainer and hasattr(trainer, 'args') else args.output_dir
        if not output_dir:
            logger.error("无法确定输出目录")
            return
        
        # 现在模型直接保存在output_dir中，不再有best_model子目录
        model_dir = Path(output_dir)
        config_file = model_dir / "config.json"
        if not config_file.exists():
            logger.error(f"模型配置文件不存在: {config_file}")
            logger.error("请确保训练完成并保存了模型")
            return
        
        # 加载训练配置
        training_config = load_training_config(output_dir)
        if training_config is None:
            logger.warning("未找到训练配置，使用当前参数")
            training_config = {}
        
        # 无论是否有trainer，都重新加载最佳模型
        logger.info(f"从 {model_dir} 重新加载最佳模型...")
        
        # 在加载模型前重新设置确定性环境，确保加载过程也是确定性的
        set_deterministic_training()
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            cache_dir="./models/clinical_modern_bert"
        )
        
        # 加载模型 - 使用训练时的配置
        dtype = torch.bfloat16 if (training_config.get('bf16', args.bf16) and torch.cuda.is_bf16_supported()) else torch.float32
        
        if os.environ.get('LOCAL_RANK'):
            # 分布式训练：不使用device_map，让DeepSpeed管理设备分配
            model = AutoModelForSequenceClassification.from_pretrained(
                str(model_dir),
                torch_dtype=dtype,
            )
            # 手动移动到当前进程的GPU
            device = f"cuda:{local_rank}"
            model = model.to(device)
            logger.info(f"模型已移动到设备: {device}")
        else:
            # 单GPU或CPU
            model = AutoModelForSequenceClassification.from_pretrained(
                str(model_dir),
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None
            )
        
        # 重建class_weights（如果训练时使用了）
        class_weights = None
        if training_config.get('use_class_weights') and training_config.get('class_weights'):
            class_weights = torch.tensor(training_config['class_weights'], dtype=torch.float32)
            logger.info(f"恢复训练时的类别权重: {class_weights}")
        
        # 创建训练参数（用于评估）- 使用训练配置确保一致性
        eval_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_eval_batch_size=training_config.get('eval_batch_size', args.batch_size * 2),
            dataloader_drop_last=False,
            eval_accumulation_steps=None,
            bf16=training_config.get('bf16', args.bf16),
            remove_unused_columns=True,
            report_to="none",  # 避免记录到wandb
            seed=training_config.get('seed', 42),
            dataloader_pin_memory=torch.cuda.is_available(),
            group_by_length=training_config.get('group_by_length', True),
        )
        
        # 创建与训练时相同的DataCollator
        eval_data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=training_config.get('pad_to_multiple_of', args.pad_to_multiple_of),
            return_tensors="pt"
        )
        
        # 创建新的BinaryClassificationTrainer用于评估，确保与训练时一致
        trainer = BinaryClassificationTrainer(
            class_weights=class_weights,  # 使用训练时的class weights以确保loss计算一致
            model=model,
            args=eval_args,
            tokenizer=tokenizer,
            data_collator=eval_data_collator,  # 使用相同的数据预处理
            compute_metrics=None  # 使用trainer内置的compute_metrics
        )
        
        logger.info("已重新创建trainer并加载最佳模型")
        logger.info(f"评估配置:")
        logger.info(f"  - batch_size: {eval_args.per_device_eval_batch_size}")
        logger.info(f"  - bf16: {eval_args.bf16}")
        logger.info(f"  - group_by_length: {eval_args.group_by_length}")
        logger.info(f"  - pad_to_multiple_of: {eval_data_collator.pad_to_multiple_of}")
        logger.info(f"  - use_class_weights: {class_weights is not None}")
        
        # 加载dev数据集
        dev_path = Path(args.data_dir) / "dev.json"
        if dev_path.exists():
            # 使用刚才加载的tokenizer
            eval_tokenizer = tokenizer
            logger.info(f"使用从 {model_dir} 加载的tokenizer")
            
            dev_dataset = ClinicalBinaryDataset(
                json_path=str(dev_path),
                tokenizer=eval_tokenizer,
                max_length=args.max_length
            )
            
            # 使用trainer进行评估（确保与训练时一致）
            logger.info("使用训练时的trainer进行验证集评估...")
            
            # 验证关键配置保持一致
            logger.info(f"评估配置验证:")
            logger.info(f"  per_device_eval_batch_size: {trainer.args.per_device_eval_batch_size}")
            logger.info(f"  bf16: {trainer.args.bf16}")
            logger.info(f"  gradient_checkpointing: {trainer.args.gradient_checkpointing}")
            logger.info(f"  data_collator.pad_to_multiple_of: {getattr(trainer.data_collator, 'pad_to_multiple_of', 'N/A')}")
            logger.info(f"  model dtype: {next(trainer.model.parameters()).dtype}")
            
            dev_metrics = trainer.evaluate(eval_dataset=dev_dataset)
            
            results['trainer_eval'] = dev_metrics
            
            # 打印评估结果
            logger.info("trainer验证集评估结果:")
            for key, value in dev_metrics.items():
                logger.info(f"  {key}: {value:.6f}")
            
            # 只在主进程保存评估结果
            if local_rank == 0:
                output_dir = trainer.args.output_dir if trainer and hasattr(trainer, 'args') else args.output_dir
                if output_dir:
                    results_path = Path(output_dir) / "trainer_evaluation_results.json"
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
                    logger.info(f"✅ Trainer评估结果保存至: {results_path}")
        else:
            logger.warning(f"未找到dev数据集: {dev_path}")
    
    
    # 执行测试集评估 - 使用trainer加载最佳模型
    if args.do_predict:
        logger.info("\n" + "="*40)
        logger.info("执行测试集评估 - 使用trainer加载最佳模型")
        logger.info(f"当前进程 LOCAL_RANK: {local_rank}")
        logger.info("="*40)
        
        # 如果没有trainer（没有执行do_eval），需要创建一个
        if 'trainer' not in locals():
            logger.info("创建用于测试集评估的trainer...")
            
            # 检查最佳模型目录
            model_dir = Path(args.output_dir) / "best_model"
            if not model_dir.exists():
                logger.error(f"最佳模型目录不存在: {model_dir}")
                return
            
            # 加载训练配置
            training_config = load_training_config(args.output_dir)
            if training_config is None:
                logger.warning("未找到训练配置，使用当前参数")
                training_config = {}
            
            # 设置确定性环境
            set_deterministic_training()
            
            # 加载tokenizer和模型
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir),
                cache_dir="./models/clinical_modern_bert"
            )
            
            dtype = torch.bfloat16 if (training_config.get('bf16', args.bf16) and torch.cuda.is_bf16_supported()) else torch.float32
            
            model = AutoModelForSequenceClassification.from_pretrained(
                str(model_dir),
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # 重建class_weights
            class_weights = None
            if training_config.get('use_class_weights') and training_config.get('class_weights'):
                class_weights = torch.tensor(training_config['class_weights'], dtype=torch.float32)
            
            # 创建评估参数
            eval_args = TrainingArguments(
                output_dir=args.output_dir,
                per_device_eval_batch_size=training_config.get('eval_batch_size', args.batch_size * 2),
                dataloader_drop_last=False,
                bf16=training_config.get('bf16', args.bf16),
                remove_unused_columns=True,
                report_to="none",
                seed=training_config.get('seed', 42),
                dataloader_pin_memory=torch.cuda.is_available(),
                group_by_length=training_config.get('group_by_length', True),
            )
            
            # 创建DataCollator
            eval_data_collator = DataCollatorWithPadding(
                tokenizer=tokenizer,
                padding=True,
                pad_to_multiple_of=training_config.get('pad_to_multiple_of', args.pad_to_multiple_of),
                return_tensors="pt"
            )
            
            # 创建trainer
            trainer = BinaryClassificationTrainer(
                class_weights=class_weights,
                model=model,
                args=eval_args,
                tokenizer=tokenizer,
                data_collator=eval_data_collator,
                compute_metrics=None
            )
        
        # 判断是否需要重新加载模型
        # 1. 如果do_eval没有执行过，需要重新加载
        # 2. 如果刚训练完且使用了LoRA，trainer.model已被merge_and_unload破坏，也需要重新加载
        # 确保report_to为None
        if trainer.args.report_to != None:
            logger.info("设置trainer的report_to为None以避免重复记录")
            trainer.args.report_to = None
        
        # 加载test数据集
        test_path = Path(args.data_dir) / "test.json"
        if test_path.exists():
            # 获取tokenizer（从trainer或重新加载）
            if hasattr(trainer, 'tokenizer'):
                test_tokenizer = trainer.tokenizer
            elif hasattr(trainer, 'processing_class'):
                test_tokenizer = trainer.processing_class
            else:
                # 从best_model目录加载tokenizer
                model_dir = Path(args.output_dir) / "best_model"
                test_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                logger.info(f"从 {model_dir} 加载tokenizer")
            
            test_dataset = ClinicalBinaryDataset(
                json_path=str(test_path),
                tokenizer=test_tokenizer,
                max_length=args.max_length
            )
            
            # 使用trainer进行测试集评估（确保与训练时一致）
            logger.info("使用训练时的trainer进行测试集评估...")
            
            # 验证关键配置保持一致
            logger.info(f"预测配置验证:")
            logger.info(f"  per_device_eval_batch_size: {trainer.args.per_device_eval_batch_size}")
            logger.info(f"  bf16: {trainer.args.bf16}")
            logger.info(f"  gradient_checkpointing: {trainer.args.gradient_checkpointing}")
            logger.info(f"  data_collator.pad_to_multiple_of: {getattr(trainer.data_collator, 'pad_to_multiple_of', 'N/A')}")
            logger.info(f"  model dtype: {next(trainer.model.parameters()).dtype}")
            
            test_metrics = trainer.evaluate(eval_dataset=test_dataset)
            
            results['trainer_test'] = test_metrics
            
            # 打印测试集评估结果
            logger.info("trainer测试集评估结果:")
            for key, value in test_metrics.items():
                logger.info(f"  {key}: {value:.6f}")
            
            # 只在主进程保存测试指标
            if local_rank == 0:
                # 使用trainer的输出目录
                save_dir = trainer.args.output_dir if trainer and hasattr(trainer, 'args') else args.output_dir
                if save_dir:
                    results_path = Path(save_dir) / "trainer_test_results.json"
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
                    logger.info(f"✅ Trainer测试集评估结果保存至: {results_path}")
                else:
                    logger.warning("无法确定保存目录，跳过保存测试结果")
        else:
            logger.warning(f"未找到test数据集: {test_path}")


if __name__ == "__main__":
    main()