"""
PEFT + DeepSpeed 二分类训练脚本
支持LoRA等参数高效微调方法与DeepSpeed分布式训练
"""

# 标准库imports
import sys
import os
import argparse
import random
import logging
import json
from pathlib import Path
from datetime import datetime

# 第三方库imports
import torch
import torch.nn as nn
import numpy as np
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.data import DataLoader, Subset

# Transformers imports
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    set_seed
)

# DeepSpeed imports
import deepspeed
from deepspeed.accelerator import get_accelerator

# PEFT imports
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not installed. Install with: pip install peft")

# 本地模块imports
sys.path.append(str(Path(__file__).parent))
from data import create_dataloaders, ClinicalBinaryDataset
from utility import (
    evaluate_dataset, 
    load_best_checkpoint, 
    load_best_checkpoint_lora, 
    predict_batch, 
    compute_metrics,
    load_best_peft_model_for_eval, 
    evaluate_dataset_simple
)

# 其他工具imports
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PEFTDeepSpeedTrainer(Trainer):
    """支持PEFT和DeepSpeed的训练器"""
    
    def __init__(self, class_weights=None, peft_config=None, *args, **kwargs):
        self.peft_config = peft_config
        self.class_weights = class_weights
        
        super().__init__(*args, **kwargs)
        
        # 重新绑定compute_metrics方法
        self.compute_metrics = self._internal_compute_metrics
        
        # 初始化固定的loss函数
        self._loss_fct = None
        self._setup_loss_function()
    
    def _setup_loss_function(self):
        """设置固定的loss函数"""
        if self.class_weights is not None:
            # 将class_weights移动到正确的设备
            if hasattr(self.args, 'local_rank') and self.args.local_rank >= 0:
                local_rank = self.args.local_rank
            else:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
            
            if torch.cuda.is_available() and local_rank >= 0:
                device = f'cuda:{local_rank}'
            else:
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                
            weights = self.class_weights.to(device, dtype=torch.float32)
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
        
        # 计算整体指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        
        # 计算每个类别的指标
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
        # 添加每个类别的指标
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
        
        # 移除labels从inputs中
        inputs_without_labels = {k: v for k, v in inputs.items() if k != "labels"}
        
        # 前向传播
        outputs = model(**inputs_without_labels)
        logits = outputs.get('logits')
        
        if labels is not None:
            # 获取模型配置
            if hasattr(model, 'module'):
                num_labels = model.module.config.num_labels
            else:
                num_labels = model.config.num_labels
            
            # 确保设备和数据类型一致性
            device = logits.device
            dtype = logits.dtype
            
            # 确保labels在正确的设备上
            if labels.device != device:
                labels = labels.to(device)
            
            if self.class_weights is not None:
                # 检查是否需要更新loss函数的权重
                current_weight = self._loss_fct.weight
                if current_weight.device != device or current_weight.dtype != dtype:
                    weights = self.class_weights.to(device=device, dtype=dtype)
                    self._loss_fct = nn.CrossEntropyLoss(weight=weights)
            
            # 使用固定的loss函数
            loss = self._loss_fct(logits.view(-1, num_labels), labels.view(-1))
        else:
            loss = None
        
        return (loss, outputs) if return_outputs else loss


def create_peft_model(model, peft_config):
    """创建PEFT模型"""
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT not available. Please install: pip install peft")
    
    logger.info("应用PEFT配置...")
    model = get_peft_model(model, peft_config)
    
    # 打印可训练参数统计
    model.print_trainable_parameters()
    
    return model


def ensure_peft_dtype_consistency(model, target_dtype):
    """确保PEFT模型的数据类型一致性"""
    logger.info(f"将整个PEFT模型转换为: {target_dtype}")
    model = model.to(dtype=target_dtype)
    return model


def load_model_and_tokenizer_peft(model_name: str,
                                  cache_dir: str = "./models/clinical_modern_bert", 
                                  num_labels: int = 2,
                                  class_weights: torch.Tensor = None,
                                  is_deepspeed: bool = False,
                                  peft_config=None,
                                  args=None):
    """加载模型和tokenizer，支持PEFT"""
    
    logger.info("加载模型和tokenizer...")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    
    # 确定数据类型
    if args and hasattr(args, 'torch_dtype') and args.torch_dtype != "auto":
        if args.torch_dtype == "bfloat16":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        elif args.torch_dtype == "float16":
            dtype = torch.float16
        elif args.torch_dtype == "float32":
            dtype = torch.float32
        else:
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    else:
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
    
    # 模型加载配置
    load_kwargs = {
        "num_labels": num_labels,
        "cache_dir": cache_dir,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": getattr(args, 'low_cpu_mem_usage', True) if args else True,
        "trust_remote_code": False,
    }
    
    # DeepSpeed训练时不使用device_map
    # 评估时也不要使用device_map="auto"，避免模型分散在多个GPU上导致设备不匹配
    if not is_deepspeed and torch.cuda.is_available() and False:  # 暂时禁用device_map
        load_kwargs["device_map"] = "auto"
    
    # 加载基础模型
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        **load_kwargs
    )
    
    # 如果不是DeepSpeed且有GPU，将模型移到单个GPU上
    if not is_deepspeed and torch.cuda.is_available():
        device = torch.device("cuda:0")
        model = model.to(device)
        logger.info(f"模型已移至设备: {device}")
    
    # 应用PEFT配置
    if peft_config is not None:
        model = create_peft_model(model, peft_config)
        # 确保数据类型一致性
        model = ensure_peft_dtype_consistency(model, dtype)
        logger.info("✅ PEFT模型创建成功，数据类型已统一")
    
    # 设置类别权重
    if class_weights is not None:
        logger.info(f"设置类别权重: {class_weights}")
        if not is_deepspeed:
            if torch.cuda.is_available():
                device = next(model.parameters()).device
                model.class_weights = class_weights.to(device)
            else:
                model.class_weights = class_weights
        else:
            model.class_weights = class_weights
    
    # 模型编译（PEFT模型可能不支持compile）
    if (args and hasattr(args, 'compile_model') and args.compile_model and 
        not is_deepspeed and peft_config is None):
        logger.info("启用torch.compile模型编译优化...")
        try:
            model = torch.compile(model)
            logger.info("✅ 模型编译成功")
        except Exception as e:
            logger.warning(f"模型编译失败: {e}")
    elif peft_config is not None and args and hasattr(args, 'compile_model') and args.compile_model:
        logger.warning("PEFT模型暂不支持torch.compile，跳过编译")
    
    logger.info(f"模型类型: {type(model).__name__}")
    logger.info(f"数据类型: {dtype}")
    
    return model, tokenizer


def create_peft_config(args):
    """根据参数创建PEFT配置"""
    if not args.use_peft or not PEFT_AVAILABLE:
        return None
    
    if args.peft_type == "lora":
        # 处理lora_target_modules参数：支持逗号分隔的字符串
        if len(args.lora_target_modules) == 1 and "," in args.lora_target_modules[0]:
            # 如果是逗号分隔的字符串，分割它
            target_modules = args.lora_target_modules[0].split(",")
            target_modules = [module.strip() for module in target_modules]
        else:
            # 如果是多个参数，直接使用
            target_modules = args.lora_target_modules
        
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,  # 训练模式
            modules_to_save=["classifier"]  # 重要：保存classifier层以避免dtype不匹配
        )
        logger.info(f"创建LoRA配置: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        logger.info(f"原始目标模块参数: {args.lora_target_modules}")
        logger.info(f"处理后的目标模块: {target_modules}")
        logger.info("包含classifier层在modules_to_save中")
        
    else:
        raise NotImplementedError(f"PEFT类型 {args.peft_type} 暂未实现")
    
    return peft_config


def calculate_class_weights(train_dataset):
    """计算类别权重"""
    label_counts = {}
    for i in range(len(train_dataset)):
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
    
    # 转换为tensor
    weights_tensor = torch.tensor([class_weights[i] for i in sorted(class_weights.keys())], 
                                dtype=torch.float32)
    
    return weights_tensor


def load_datasets(data_dir: str, tokenizer, max_length: int = 512, train_ratio: float = 1.0):
    """加载训练、验证、测试数据集"""
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
            random.seed(42)
            total_samples = len(full_train_dataset)
            sample_size = max(1, int(total_samples * train_ratio))
            
            indices = list(range(total_samples))
            random.shuffle(indices)
            sampled_indices = indices[:sample_size]
            
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


def save_peft_model(trainer, output_dir, merge_and_save=False):
    """保存PEFT模型"""
    try:
        logger.info("开始保存PEFT模型...")
        
        if hasattr(trainer.model, 'save_pretrained'):
            # 保存adapter
            adapter_path = Path(output_dir) / "peft_adapter"
            adapter_path.mkdir(exist_ok=True)
            logger.info(f"保存adapter到: {adapter_path}")
            trainer.model.save_pretrained(str(adapter_path))
            logger.info(f"✅ PEFT adapter已保存至: {adapter_path}")
            
            # 如果需要，合并并保存完整模型
            if merge_and_save:
                logger.info("开始合并adapter并保存完整模型...")
                try:
                    merged_model = trainer.model.merge_and_unload()
                    merged_path = Path(output_dir) / "merged_model"
                    merged_path.mkdir(exist_ok=True)
                    logger.info(f"保存合并模型到: {merged_path}")
                    merged_model.save_pretrained(str(merged_path))
                    logger.info(f"✅ 合并后的完整模型已保存至: {merged_path}")
                except Exception as e:
                    logger.warning(f"合并模型失败，但adapter已保存: {e}")
        else:
            logger.warning("模型没有save_pretrained方法，跳过PEFT保存")
            
    except Exception as e:
        logger.error(f"保存PEFT模型时出错: {e}")
        # 尝试基本的模型保存
        try:
            trainer.save_model(output_dir)
            logger.info("已使用基础方法保存模型")
        except Exception as e2:
            logger.error(f"基础保存也失败: {e2}")
    
    logger.info("PEFT模型保存流程完成")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PEFT + DeepSpeed 二分类微调")
    
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
    parser.add_argument("--batch_size", type=int, default=4,
                        help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--max_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="预热比例")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="权重衰减")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", 
                                 "constant", "constant_with_warmup", "inverse_sqrt", "reduce_lr_on_plateau"],
                        help="学习率调度器类型（注意：使用此参数需要配合deepspeed_config_bf16_no_scheduler.json）")
    
    # 优化设置
    parser.add_argument("--pad_to_multiple_of", type=int, default=64,
                        help="padding到此倍数")
    parser.add_argument("--use_class_weights", action="store_true",
                        help="使用类别权重处理数据不平衡")
    parser.add_argument("--bf16", action="store_true",
                        help="启用bf16混合精度训练")
    parser.add_argument("--fp16", action="store_true",
                        help="启用fp16混合精度训练")
    
    # 性能优化选项
    parser.add_argument("--compile_model", action="store_true",
                        help="使用torch.compile编译模型")
    parser.add_argument("--torch_dtype", type=str, default="auto",
                        choices=["auto", "float32", "float16", "bfloat16"],
                        help="模型权重数据类型")
    parser.add_argument("--low_cpu_mem_usage", action="store_true", default=True,
                        help="启用低CPU内存使用模式")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="启用梯度检查点")
    
    # PEFT选项
    parser.add_argument("--use_peft", action="store_true",
                        help="启用PEFT（参数高效微调）")
    parser.add_argument("--peft_type", type=str, default="lora",
                        choices=["lora"],
                        help="PEFT方法类型")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank值")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha缩放参数")
    parser.add_argument("--lora_dropout", type=float, default=0.4,
                        help="LoRA dropout率")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", 
                        default=["query", "value"],
                        help="LoRA目标模块（可以传入逗号分隔的字符串或多个参数）")
    parser.add_argument("--merge_and_save_peft", action="store_true",
                        help="训练完成后合并adapter并保存完整模型")
    
    # DeepSpeed设置
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="DeepSpeed配置文件路径")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="分布式训练的local rank")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--train_ratio", type=float, default=1.0,
                        help="训练集使用比例")
    parser.add_argument("--report_to", type=str, default="none",
                        choices=["none", "wandb", "tensorboard", "all"],
                        help="日志报告工具")
    
    # 任务控制
    parser.add_argument("--do_train", action="store_true",
                        help="是否执行训练")
    parser.add_argument("--do_eval", action="store_true",
                        help="是否执行dev集评估")
    parser.add_argument("--do_predict", action="store_true",
                        help="是否执行test集评估")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="已训练模型的checkpoint路径，用于直接评估/预测")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # 获取分布式训练配置
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank == 0:
        logger.info("="*60)
        logger.info("PEFT + DeepSpeed 二分类训练")
        logger.info(f"World Size: {world_size}, Local Rank: {local_rank}")
        if args.use_peft:
            logger.info(f"PEFT启用: {args.peft_type}")
            if args.peft_type == "lora":
                logger.info(f"LoRA配置: r={args.lora_r}, alpha={args.lora_alpha}")
        logger.info("="*60)
    
    # 自动生成输出目录（只在主进程生成，避免多进程竞争）
    if args.output_dir:
        if local_rank == 0:
            # 在指定的目录下创建带时间戳的子目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short_name = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
            peft_suffix = f"_peft_{args.peft_type}_r{args.lora_r}" if args.use_peft else ""
            # 确保不会创建嵌套的相对路径
            base_dir = Path(args.output_dir).resolve()  # 转换为绝对路径
            args.output_dir = str(base_dir / f"{model_short_name}_{timestamp}{peft_suffix}")
            # 创建目录
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            # 将output_dir写入临时文件，供其他进程读取
            with open("/tmp/output_dir.txt", "w") as f:
                f.write(args.output_dir)
        else:
            # 非主进程等待主进程创建目录并读取
            import time
            time.sleep(2)  # 等待主进程创建目录
            try:
                with open("/tmp/output_dir.txt", "r") as f:
                    args.output_dir = f.read().strip()
            except FileNotFoundError:
                # 如果读取失败，使用fallback方案
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_short_name = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
                peft_suffix = f"_peft_{args.peft_type}_r{args.lora_r}" if args.use_peft else ""
                args.output_dir = f"./outputs/{model_short_name}_{timestamp}{peft_suffix}"
    else:
        # 如果没有指定输出目录，设置默认值并在主进程创建
        if local_rank == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_short_name = args.model_name.split('/')[-1] if '/' in args.model_name else args.model_name
            peft_suffix = f"_peft_{args.peft_type}_r{args.lora_r}" if args.use_peft else ""
            args.output_dir = f"./outputs/{model_short_name}_{timestamp}{peft_suffix}"
            # 创建目录
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"未指定输出目录，使用默认值: {args.output_dir}")
    if local_rank == 0:
        logger.info(f"输出目录: {args.output_dir}")
    
    # 根据是否训练决定PEFT配置
    if args.do_train:
        # 训练模式：创建PEFT配置
        peft_config = create_peft_config(args)
    else:
        # 评估/预测模式：不创建PEFT配置，稍后根据checkpoint类型决定
        peft_config = None
    
    # 加载模型和tokenizer
    model, tokenizer = load_model_and_tokenizer_peft(
        model_name=args.model_name,
        cache_dir="./models/clinical_modern_bert",
        is_deepspeed=args.deepspeed is not None,
        peft_config=peft_config,
        args=args
    )
    
    # 如果提供了checkpoint路径，加载已训练的模型
    if args.checkpoint_path and not args.do_train:
        logger.info(f"加载checkpoint: {args.checkpoint_path}")
        
        # 检查是否是PEFT adapter目录
        is_peft_adapter = os.path.exists(os.path.join(args.checkpoint_path, "adapter_config.json"))
        
        if is_peft_adapter and args.use_peft:
            # PEFT模型加载
            model = PeftModel.from_pretrained(model, args.checkpoint_path)
            logger.info("✅ PEFT adapter checkpoint加载成功")
            
            # 如果是best_checkpoint目录，特别标注
            if "best_checkpoint" in args.checkpoint_path:
                logger.info("📌 已加载最佳模型checkpoint")
        else:
            # 普通模型或合并后的模型加载
            checkpoint_model = AutoModelForSequenceClassification.from_pretrained(
                args.checkpoint_path,
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                device_map="auto" if torch.cuda.is_available() and not args.deepspeed else None
            )
            model = checkpoint_model
            logger.info("✅ 完整模型checkpoint加载成功")
            
            # 如果是best_checkpoint或merged_model目录，特别标注
            if "best_checkpoint" in args.checkpoint_path:
                logger.info("📌 已加载最佳模型checkpoint")
            elif "merged_model" in args.checkpoint_path:
                logger.info("📌 已加载合并后的完整模型")
        
        # 在非训练模式下，确保模型在正确的设备上
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
            model = model.to(device)
            logger.info(f"模型已移至设备: {device}")
    elif args.checkpoint_path and args.do_train:
        logger.warning("checkpoint_path参数仅在非训练模式下生效（不使用--do_train）")
    
    # 加载数据集
    if local_rank == 0:
        logger.info("加载数据集...")
    datasets = load_datasets(args.data_dir, tokenizer, args.max_length, args.train_ratio)
    
    train_dataset = datasets.get('train')
    dev_dataset = datasets.get('dev')
    test_dataset = datasets.get('test')
    
    if not train_dataset:
        raise ValueError("训练数据集未找到")
    
    # 计算类别权重
    class_weights = None
    if args.use_class_weights:
        if local_rank == 0:
            logger.info("计算类别权重...")
        class_weights = calculate_class_weights(train_dataset)
    
    # 创建DataCollator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=args.pad_to_multiple_of,
        return_tensors="pt"
    )
    
    # 创建训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        
        # 评估策略
        eval_strategy="epoch" if dev_dataset else "no",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True if dev_dataset else False,
        metric_for_best_model="eval_f1_class_1" if dev_dataset else None,
        greater_is_better=True if dev_dataset else None,
        
        # 日志设置
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        logging_first_step=True,
        report_to=args.report_to if local_rank == 0 else "none",
        
        # 优化设置
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_pin_memory=torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        group_by_length=True,
        
        # DeepSpeed设置
        deepspeed=args.deepspeed,
        local_rank=local_rank,
        
        # 其他设置
        seed=args.seed,
        push_to_hub=False,
        remove_unused_columns=True,
    )
    
    # 创建回调函数
    callbacks = []
    if dev_dataset:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=10))
    
    # 创建Trainer
    trainer = PEFTDeepSpeedTrainer(
        class_weights=class_weights,
        peft_config=peft_config,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # 开始训练
    if args.do_train:
        if local_rank == 0:
            logger.info("开始PEFT + DeepSpeed训练...")
        
        logger.info("调用trainer.train()...")
        train_result = trainer.train()
        
        # 强制立即输出日志
        print("=== trainer.train()完成 ===", flush=True)
        sys.stdout.flush()
        logger.info("trainer.train()完成")
        
        print(f"=== 训练结果类型: {type(train_result)} ===", flush=True)
        sys.stdout.flush()
        logger.info(f"训练结果类型: {type(train_result)}")
        
        print("=== 训练阶段完全结束 ===", flush=True)
        sys.stdout.flush()
        logger.info("训练阶段完全结束")
        
        # 保存模型
        print("=== 开始保存模型流程 ===", flush=True)
        sys.stdout.flush()
        if local_rank == 0:
            print("=== 主进程开始保存 ===", flush=True)
            sys.stdout.flush()
            logger.info("开始保存模型和tokenizer...")
            print("=== logger.info 完成 ===", flush=True)
            sys.stdout.flush()
            
            # 保存最佳checkpoint（load_best_model_at_end=True时，当前模型就是最佳模型）
            best_checkpoint_dir = Path(args.output_dir) / "best_checkpoint"
            print(f"=== 准备保存到: {best_checkpoint_dir} ===", flush=True)
            sys.stdout.flush()
            logger.info(f"保存最佳模型到: {best_checkpoint_dir}")
            print("=== checkpoint目录log完成 ===", flush=True)
            sys.stdout.flush()
            
            if args.use_peft:
                print("=== 开始PEFT保存流程 ===", flush=True)
                sys.stdout.flush()
                # 保存PEFT最佳模型
                print("=== 创建目录 ===", flush=True)
                sys.stdout.flush()
                best_checkpoint_dir.mkdir(exist_ok=True, parents=True)
                print("=== 目录创建完成，开始save_pretrained ===", flush=True)
                sys.stdout.flush()
                
                # 添加超时保护
                import signal
                
                
                try:
                    trainer.model.save_pretrained(str(best_checkpoint_dir))
                    print("=== save_pretrained完成 ===", flush=True)
                    sys.stdout.flush()
                    logger.info(f"✅ 最佳PEFT模型已保存至: {best_checkpoint_dir}")
                except TimeoutError:
                    print("=== save_pretrained超时，跳过 ===", flush=True)
                    logger.warning("save_pretrained 超时，跳过模型保存")
                finally:
                    signal.alarm(0)  # 取消超时
                
                # 也保存到常规目录
                print("=== 调用save_peft_model ===", flush=True)
                save_peft_model(trainer, args.output_dir, args.merge_and_save_peft)
                print("=== save_peft_model完成 ===", flush=True)
            else:
                # 保存普通最佳模型
                trainer.save_model(str(best_checkpoint_dir))
                logger.info(f"✅ 最佳模型已保存至: {best_checkpoint_dir}")
                
                # 也保存到主目录
                trainer.save_model(args.output_dir)
            
            # 保存tokenizer到最佳checkpoint目录
            tokenizer.save_pretrained(str(best_checkpoint_dir))
            
            print("=== 保存tokenizer ===", flush=True)
            logger.info("保存tokenizer...")
            tokenizer.save_pretrained(args.output_dir)
            print("=== tokenizer保存完成 ===", flush=True)
            logger.info("✅ 训练和保存完成！")
        
        print("=== 保存流程完全结束 ===", flush=True)
    
    # 仅评估模式 - 不训练，只从checkpoint加载并评估
    if (args.do_eval or args.do_predict) and local_rank == 0 and not args.do_train:
        logger.info("=" * 60)
        logger.info("开始仅评估模式...")
        logger.info("=" * 60)
        
        if not args.checkpoint_path:
            logger.error("仅评估模式需要提供--checkpoint_path参数")
            return
        
        try:
            # 直接使用已加载的模型进行评估
            eval_model = model
            eval_tokenizer = tokenizer
            
            # 验证集评估
            if args.do_eval and dev_dataset:
                logger.info("开始验证集评估...")
                dev_results = evaluate_dataset_simple(
                    model=eval_model,
                    tokenizer=eval_tokenizer,
                    dataset=dev_dataset,
                    batch_size=args.batch_size * 2
                )
                
                logger.info("验证集评估结果:")
                for key, value in dev_results.items():
                    logger.info(f"  {key}: {value:.4f}")
            
            # 测试集评估
            if args.do_predict and test_dataset:
                logger.info("\n开始测试集评估...")
                test_results = evaluate_dataset_simple(
                    model=eval_model,
                    tokenizer=eval_tokenizer,
                    dataset=test_dataset,
                    batch_size=args.batch_size * 2
                )
                
                logger.info("测试集评估结果:")
                for key, value in test_results.items():
                    logger.info(f"  {key}: {value:.4f}")
            
            logger.info("\n✅ 仅评估模式完成！")
            
        except Exception as e:
            logger.error(f"❌ 评估失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # 仅评估模式完成后，立即退出避免分布式清理问题
        if not args.do_train:
            logger.info("仅评估模式结束，直接退出")
            sys.exit(0)
    
    # 评估和测试 - 只在主进程执行，且在训练完成后
    elif (args.do_eval or args.do_predict) and local_rank == 0 and args.do_train:
        logger.info("=" * 60)
        logger.info("开始训练后评估...")
        logger.info("=" * 60)
        
        # 不需要barrier同步，因为只有主进程执行评估
        logger.info("主进程开始训练后评估（其他进程已完成）...")
        
        # 使用已导入的评估函数
        # 添加超时保护，避免评估阶段卡住

        try:
            # 释放训练时的模型内存
            if args.deepspeed:
                del trainer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("已释放训练模型内存")
            
            # 加载最佳模型进行评估
            best_checkpoint_path = Path(args.output_dir) / "best_checkpoint"
            if best_checkpoint_path.exists():
                eval_model, eval_tokenizer = load_best_peft_model_for_eval(
                    checkpoint_path=str(best_checkpoint_path),
                    model_name=args.model_name
                )
                
                # 验证集评估
                if args.do_eval and dev_dataset:
                    logger.info("开始验证集评估...")
                    print("=== 调用evaluate_dataset_simple - dev ===", flush=True)
                    dev_results = evaluate_dataset_simple(
                        model=eval_model,
                        tokenizer=eval_tokenizer,
                        dataset=dev_dataset,
                        batch_size=min(args.batch_size * 2, 16)  # 限制batch size避免内存问题
                    )
                    print("=== evaluate_dataset_simple - dev 完成 ===", flush=True)
                    logger.info("验证集评估函数已返回，处理结果...")
                    
                    logger.info("验证集评估结果:")
                    for key, value in dev_results.items():
                        if isinstance(value, (int, float)):
                            logger.info(f"  {key}: {value:.4f}")
                    
                    # 记录到wandb
                    if args.report_to in ["wandb", "all"]:
                        try:
                            if wandb.run is not None:
                                best_eval_metrics = {f"best_eval/{k}": v for k, v in dev_results.items() if isinstance(v, (int, float))}
                                wandb.log(best_eval_metrics)
                                summary_metrics = {f"final_best_eval_{k}": v for k, v in dev_results.items() if isinstance(v, (int, float))}
                                wandb.summary.update(summary_metrics)
                                logger.info("✅ 验证集评估结果已记录到 wandb")
                        except Exception as e:
                            logger.warning(f"记录到 wandb 失败: {e}")
                
                # 测试集预测
                if args.do_predict and test_dataset:
                    logger.info("开始测试集评估...")
                    print("=== 调用evaluate_dataset_simple - test ===", flush=True)
                    test_results = evaluate_dataset_simple(
                        model=eval_model,
                        tokenizer=eval_tokenizer,
                        dataset=test_dataset,
                        batch_size=min(args.batch_size * 2, 16)  # 限制batch size避免内存问题
                    )
                    print("=== evaluate_dataset_simple - test 完成 ===", flush=True)
                    
                    logger.info("测试集评估结果:")
                    for key, value in test_results.items():
                        if isinstance(value, (int, float)):
                            logger.info(f"  {key}: {value:.4f}")
                    
                    # 记录到wandb
                    if args.report_to in ["wandb", "all"]:
                        try:
                            if wandb.run is not None:
                                test_metrics = {f"test/{k}": v for k, v in test_results.items() if isinstance(v, (int, float))}
                                wandb.log(test_metrics)
                                wandb.summary.update({f"final_{k}": v for k, v in test_results.items() if isinstance(v, (int, float))})
                                logger.info("✅ 测试集评估结果已记录到 wandb")
                        except Exception as e:
                            logger.warning(f"记录到 wandb 失败: {e}")
                
                # 清理评估模型
                del eval_model, eval_tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            else:
                logger.warning(f"最佳checkpoint不存在: {best_checkpoint_path}")
                
        except TimeoutError:
            logger.warning("训练后评估超时，跳过评估步骤")
        except Exception as e:
            logger.error(f"评估过程出错: {e}")
            logger.error(traceback.format_exc())
        finally:
            signal.alarm(0)  # 取消超时
        
        logger.info("=" * 60)
        logger.info("评估完成")
        logger.info("=" * 60)
    
    

    logger.info("程序即将结束")
    logger.info("="*50)
    logger.info("训练流程完全结束")


if __name__ == "__main__":
    main()