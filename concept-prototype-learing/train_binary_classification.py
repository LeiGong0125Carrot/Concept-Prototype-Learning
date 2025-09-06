"""
使用AutoModelForSequenceClassification和Trainer进行二分类微调
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np
import argparse
import random
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging
import json
from data import create_dataloaders, ClinicalBinaryDataset
from torch.utils.data import DataLoader, Subset
from utility import evaluate_dataset, load_best_checkpoint, predict_batch

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


def load_model_and_tokenizer(model_name: str, 
                           cache_dir: str = "./models/clinical_modern_bert",
                           num_labels: int = 2,
                           class_weights: torch.Tensor = None):
    """
    加载模型和tokenizer
    """
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
        "torch_dtype": dtype,
        "ignore_mismatched_sizes": True
    }
    
    # 只有在非DeepSpeed情况下才设置device_map
    # DeepSpeed会自己管理设备分配
    import os
    if not os.environ.get('LOCAL_RANK') and torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        **load_kwargs
    )
    
    # 如果有class weights，设置到模型中（可选：修改loss function）
    if class_weights is not None:
        logger.info(f"设置类别权重: {class_weights}")
        # 可以将权重保存到model以供后续使用
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
    
    def _internal_compute_metrics(self, eval_pred):
        """计算训练过程中的评估指标"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        
        # 计算整体指标（binary average）
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )
        
        # 计算每个类别的F1分数
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            labels, predictions, average=None
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
            metrics[f'recall_class_{i}'] = recall_score if not np.isnan(f1_score) else 0.0
        
        for i, precision_score in enumerate(precision_per_class):
            metrics[f'recall_class_{i}'] = precision_score if not np.isnan(f1_score) else 0.0
        
        return metrics
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """自定义损失函数支持类别权重"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        if labels is not None and self.class_weights is not None:
            # 获取模型配置 - 处理DistributedDataParallel包装的情况
            if hasattr(model, 'module'):
                # 分布式训练时模型被包装在DistributedDataParallel中
                num_labels = model.module.config.num_labels
            else:
                # 单卡训练
                num_labels = model.config.num_labels
            
            # 使用weighted cross entropy loss
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(self.args.device))
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        else:
            # 使用模型默认loss
            loss = outputs.loss
        
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
            metric_for_best_model="f1",
            greater_is_better=True,
            
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


def train_clinical_model(
    model_name: str = "Simonlee711/Clinical_ModernBERT",
    data_dir: str = "./data",
    output_dir: str = "./output/binary_classification",
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
    deepspeed_config: str = None
):
    """统一的训练函数"""
    
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
        load_best_model_at_end=True if dev_dataset else False,
        metric_for_best_model="eval_f1_class_1" if dev_dataset else None,  # 使用少数类F1分数
        greater_is_better=True if dev_dataset else None,
        
        # 日志设置
        logging_dir=f"{output_dir}/logs",
        logging_steps=5,
        report_to=report_to,
        
        # 优化设置 - DeepSpeed会覆盖这些设置
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported() if not deepspeed_config else False,
        dataloader_pin_memory=torch.cuda.is_available(),
        gradient_checkpointing=False,
        group_by_length=True,
        
        # DeepSpeed设置
        deepspeed=deepspeed_config,
        
        # 其他设置
        seed=42,
        push_to_hub=False,
        remove_unused_columns=True
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
    
    # 8. 开始训练
    logger.info("开始训练...")
    logger.info(f"训练参数: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}")
    logger.info(f"类别权重: {'启用' if use_class_weights else '禁用'}")
    
    train_result = trainer.train()
    
    # 9. 保存模型
    # 当 load_best_model_at_end=True 时，trainer.model 已经是最佳模型
    # 保存到默认目录（兼容旧代码）
    trainer.save_model()
    trainer.save_state()
    
    # 额外保存最佳模型到专门的目录
    if training_args.load_best_model_at_end:
        best_model_dir = Path(output_dir) / "best_model"
        logger.info(f"保存最佳模型到专门目录: {best_model_dir}")
        trainer.save_model(str(best_model_dir))  # 直接传路径，不使用output_path参数
        # 同时保存tokenizer
        trainer.tokenizer.save_pretrained(str(best_model_dir))
        
        # 保存最佳模型的元信息
        if hasattr(trainer.state, 'best_metric'):
            best_model_info = {
                "best_metric": trainer.state.best_metric,
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
                "metric_for_best_model": training_args.metric_for_best_model,
            }
            with open(best_model_dir / "best_model_info.json", 'w') as f:
                json.dump(best_model_info, f, indent=2)
    
    # 10. 评估和保存结果
    results = {'train': train_result.metrics}
    
    if dev_dataset:
        logger.info("评估验证集...")
        eval_result = trainer.evaluate()
        results['eval'] = eval_result
        
        logger.info("验证集结果:")
        for key, value in eval_result.items():
            if key.startswith('eval_'):
                logger.info(f"  {key}: {value:.4f}")
    
    # 测试集评估（训练后自动评估）
    test_dataset = datasets.get('test')
    if test_dataset:
        logger.info("使用训练后的模型评估测试集...")
        test_metrics = evaluate_dataset(trainer.model, trainer.tokenizer, test_dataset)
        results['test'] = test_metrics
    
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
    parser.add_argument("--output_dir", type=str, default="./output/binary_classification",
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
    parser.add_argument("--do_train", action="store_true", default=True,
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
    
    # 其他
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--train_ratio", type=float, default=1.0,
                        help="训练集使用比例 (0.0, 1.0]，用于快速测试功能")
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
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
            deepspeed_config=args.deepspeed
        )
        results.update(train_results)
    
    # 执行评估（仅评估dev dataset）
    if args.do_eval:
        logger.info("\n" + "="*40)
        logger.info("执行评估任务 (Dev Dataset)")
        logger.info("="*40)
        
        output_dir = args.output_dir
        
        # 加载最佳检查点
        logger.info("加载最佳模型检查点...")
        model, tokenizer = load_best_checkpoint(
            output_dir=output_dir,
            model_name=args.model_name,
            cache_dir="./models/clinical_modern_bert"
        )
        
        # 加载dev数据集
        dev_path = Path(args.data_dir) / "dev.json"
        if dev_path.exists():
            dev_dataset = ClinicalBinaryDataset(
                json_path=str(dev_path),
                tokenizer=tokenizer,
                max_length=args.max_length
            )
            
            # 评估
            dev_metrics = evaluate_dataset(
                model=model,
                tokenizer=tokenizer,
                dataset=dev_dataset,
                batch_size=min(64, args.batch_size * 8)  # 增加评估批次大小
            )
            results['dev_eval'] = dev_metrics
            
            # 保存评估结果
            results_path = Path(output_dir) / "dev_evaluation_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
            logger.info(f"✅ Dev评估结果保存至: {results_path}")
        else:
            logger.warning(f"未找到dev数据集: {dev_path}")
    
    # 执行测试集评估
    if args.do_predict:
        logger.info("\n" + "="*40)
        logger.info("执行测试集评估")
        logger.info("="*40)
        
        output_dir = args.output_dir
        
        # 加载最佳检查点
        logger.info("加载最佳模型检查点...")
        model, tokenizer = load_best_checkpoint(
            output_dir=output_dir,
            model_name=args.model_name,
            cache_dir="./models/clinical_modern_bert"
        )
        
        # 加载test数据集
        test_path = Path(args.data_dir) / "test.json"
        if test_path.exists():
            test_dataset = ClinicalBinaryDataset(
                json_path=str(test_path),
                tokenizer=tokenizer,
                max_length=args.max_length
            )
            
            # 评估测试集（返回详细预测结果）
            test_metrics = evaluate_dataset(
                model=model,
                tokenizer=tokenizer,
                dataset=test_dataset,
                batch_size=min(64, args.batch_size * 8),  # 增加评估批次大小以加速
                return_predictions=True  # 返回每个样本的预测结果
            )
            
            # 分离指标和预测结果
            predictions_detail = test_metrics.pop('predictions', None)
            results['test_eval'] = test_metrics
            
            # 保存测试指标
            results_path = Path(output_dir) / "test_evaluation_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
            logger.info(f"✅ Test评估结果保存至: {results_path}")
            
            # 保存详细预测结果（用于case study）
            if predictions_detail:
                predictions_path = Path(output_dir) / "test_predictions_detail.json"
                with open(predictions_path, 'w') as f:
                    json.dump({
                        'summary': test_metrics,
                        'predictions': predictions_detail
                    }, f, indent=2)
                logger.info(f"✅ 详细预测结果保存至: {predictions_path} (用于case study)")
                
                # 统计错误预测
                errors = [p for p in predictions_detail if not p['correct']]
                logger.info(f"错误预测数量: {len(errors)}/{len(predictions_detail)} ({len(errors)/len(predictions_detail)*100:.2f}%)")
        else:
            logger.warning(f"未找到test数据集: {test_path}")


if __name__ == "__main__":
    main()