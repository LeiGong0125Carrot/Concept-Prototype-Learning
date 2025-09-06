"""
Clinical ModernBERT 使用 Hugging Face Trainer 进行训练
支持分类任务和Token Embedding提取
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 模型配置
MODEL_NAME = "Simonlee711/Clinical_ModernBERT"
PROJECT_BASE = Path("/bigtemp/nkw3mr/concept-prototype-learing")
MODEL_CACHE_DIR = PROJECT_BASE / "models" / "clinical_modern_bert"
OUTPUT_DIR = PROJECT_BASE / "output"


class ClinicalTextDataset(Dataset):
    """临床文本数据集，适配Trainer"""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Trainer会使用DataCollator进行动态padding
            return_tensors=None
        )
        
        encoding['labels'] = label
        return encoding


@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: str = MODEL_NAME
    cache_dir: Optional[str] = str(MODEL_CACHE_DIR)
    num_labels: int = 2
    max_seq_length: int = 512
    

@dataclass
class DataArguments:
    """数据相关参数"""
    train_file: Optional[str] = None
    validation_file: Optional[str] = None
    test_file: Optional[str] = None
    text_column: str = "text"
    label_column: str = "label"


class ClinicalModernBERTTrainer:
    """使用Trainer API的Clinical ModernBERT训练器"""
    
    def __init__(
        self,
        model_args: Optional[ModelArguments] = None,
        training_args: Optional[TrainingArguments] = None
    ):
        """
        初始化训练器
        
        Args:
            model_args: 模型参数
            training_args: 训练参数
        """
        self.model_args = model_args or ModelArguments()
        self.training_args = training_args
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=True
        )
        
        # 如果tokenizer没有pad token，添加一个
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = None
        self.trainer = None
    
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        
        # 如果是logits，取argmax
        if len(predictions.shape) > 1:
            predictions = np.argmax(predictions, axis=1)
        
        # 计算指标
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train_classifier(
        self,
        train_texts: List[str],
        train_labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        num_labels: Optional[int] = None,
        training_args: Optional[TrainingArguments] = None,
        use_early_stopping: bool = True,
        early_stopping_patience: int = 3
    ):
        """
        训练分类器
        
        Args:
            train_texts: 训练文本
            train_labels: 训练标签
            val_texts: 验证文本
            val_labels: 验证标签
            num_labels: 类别数（如果None则自动推断）
            training_args: 训练参数
            use_early_stopping: 是否使用早停
            early_stopping_patience: 早停patience
        """
        # 推断类别数
        if num_labels is None:
            num_labels = len(set(train_labels))
            if val_labels:
                num_labels = max(num_labels, len(set(val_labels)))
        
        logger.info(f"类别数: {num_labels}")
        
        # 加载模型
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=self.model_args.cache_dir,
            ignore_mismatched_sizes=True
        )
        
        # 创建数据集
        train_dataset = ClinicalTextDataset(
            train_texts, 
            train_labels, 
            self.tokenizer,
            self.model_args.max_seq_length
        )
        
        eval_dataset = None
        if val_texts and val_labels:
            eval_dataset = ClinicalTextDataset(
                val_texts,
                val_labels,
                self.tokenizer,
                self.model_args.max_seq_length
            )
        
        # 默认训练参数
        if training_args is None:
            training_args = TrainingArguments(
                output_dir=str(OUTPUT_DIR / "classifier"),
                num_train_epochs=3,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=16,
                learning_rate=2e-5,
                warmup_ratio=0.1,
                weight_decay=0.01,
                logging_dir=str(OUTPUT_DIR / "logs"),
                logging_steps=10,
                eval_strategy="epoch" if eval_dataset else "no",
                save_strategy="epoch",
                save_total_limit=2,
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model="f1" if eval_dataset else None,
                greater_is_better=True,
                fp16=torch.cuda.is_available(),
                gradient_checkpointing=False,
                report_to="none",
                seed=42
            )
        
        # 数据整理器（动态padding）
        data_collator = DataCollatorWithPadding(
            self.tokenizer,
            padding=True
        )
        
        # 回调函数
        callbacks = []
        if use_early_stopping and eval_dataset:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience
                )
            )
        
        # 创建Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks
        )
        
        # 训练
        logger.info("开始训练...")
        train_result = self.trainer.train()
        
        # 保存模型
        self.trainer.save_model()
        
        # 保存训练指标
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        # 评估
        if eval_dataset:
            logger.info("评估模型...")
            eval_metrics = self.trainer.evaluate()
            self.trainer.log_metrics("eval", eval_metrics)
            self.trainer.save_metrics("eval", eval_metrics)
            
            return train_result, eval_metrics
        
        return train_result, None
    
    def predict(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测
        
        Args:
            texts: 待预测文本
            batch_size: 批次大小
            
        Returns:
            (predictions, probabilities)
        """
        if self.trainer is None:
            raise ValueError("请先训练模型或加载已训练的模型")
        
        # 创建预测数据集
        dummy_labels = [0] * len(texts)
        predict_dataset = ClinicalTextDataset(
            texts,
            dummy_labels,
            self.tokenizer,
            self.model_args.max_seq_length
        )
        
        # 预测
        predictions = self.trainer.predict(
            predict_dataset,
            metric_key_prefix="predict"
        )
        
        # 获取预测结果和概率
        logits = predictions.predictions
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        preds = np.argmax(logits, axis=1)
        
        return preds, probs
    
    def load_model(self, model_path: str):
        """加载已训练的模型"""
        logger.info(f"从 {model_path} 加载模型...")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 创建一个简单的Trainer用于预测
        self.trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        logger.info("模型加载成功")


class ClinicalEmbeddingExtractor:
    """Clinical ModernBERT Embedding提取器"""
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        cache_dir: Optional[Path] = MODEL_CACHE_DIR,
        device: Optional[str] = None
    ):
        """初始化Embedding提取器"""
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"初始化Embedding提取器 - 设备: {self.device}")
        
        # 加载模型和tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        ).to(self.device)
        
        self.model.eval()
    
    def extract_embeddings(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        max_length: int = 512,
        pooling_strategy: str = 'mean',
        return_tensors: bool = False,
        show_progress: bool = True
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        批量提取文本embeddings
        
        Args:
            texts: 输入文本
            batch_size: 批次大小
            max_length: 最大序列长度
            pooling_strategy: 池化策略 ('cls', 'mean', 'max')
            return_tensors: 是否返回PyTorch张量
            show_progress: 是否显示进度条
            
        Returns:
            文本embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        all_embeddings = []
        
        # 批处理
        from tqdm import tqdm
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Extracting embeddings")
        
        for i in iterator:
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # 提取embeddings
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                hidden_states = outputs.last_hidden_state
                
                # 应用池化策略
                if pooling_strategy == 'cls':
                    embeddings = hidden_states[:, 0, :]
                
                elif pooling_strategy == 'mean':
                    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    masked_embeddings = hidden_states * mask
                    summed = torch.sum(masked_embeddings, dim=1)
                    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
                    embeddings = summed / counts
                
                elif pooling_strategy == 'max':
                    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    masked_embeddings = hidden_states * mask + (1 - mask) * (-1e9)
                    embeddings = torch.max(masked_embeddings, dim=1)[0]
                
                else:
                    raise ValueError(f"不支持的池化策略: {pooling_strategy}")
                
                all_embeddings.append(embeddings.cpu())
        
        # 合并所有批次
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        if return_tensors:
            return all_embeddings
        else:
            return all_embeddings.numpy()
    
    def get_token_embeddings(
        self,
        text: str,
        max_length: int = 512,
        return_tokens: bool = True
    ) -> Dict:
        """
        获取单个文本的token级embeddings
        
        Args:
            text: 输入文本
            max_length: 最大序列长度
            return_tokens: 是否返回token列表
            
        Returns:
            包含embeddings和可选tokens的字典
        """
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # 提取embeddings
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            embeddings = outputs.last_hidden_state[0]  # 移除batch维度
        
        # 只保留非padding的tokens
        valid_length = attention_mask[0].sum().item()
        embeddings = embeddings[:valid_length]
        
        result = {
            'embeddings': embeddings.cpu().numpy(),
            'shape': embeddings.shape
        }
        
        if return_tokens:
            tokens = self.tokenizer.convert_ids_to_tokens(
                input_ids[0][:valid_length].cpu().numpy()
            )
            result['tokens'] = tokens
        
        return result


# 使用示例
def example_trainer_classification():
    """使用Trainer的分类示例"""
    logger.info("="*60)
    logger.info("使用Trainer API进行分类任务")
    logger.info("="*60)
    
    # 准备示例数据
    train_texts = [
        "Patient presents with acute chest pain and elevated troponin levels.",
        "Blood glucose 250 mg/dL, patient has polyuria and polydipsia.",
        "MRI shows multiple white matter lesions consistent with MS.",
        "Hemoglobin A1c is 9.5%, indicating poor glycemic control.",
        "ECG shows ST elevation in leads II, III, and aVF.",
        "Patient has progressive weakness and fasciculations."
    ]
    
    # 标签: 0=心血管, 1=内分泌, 2=神经
    train_labels = [0, 1, 2, 1, 0, 2]
    
    val_texts = [
        "Cardiac enzymes are elevated.",
        "Fasting glucose is 180 mg/dL."
    ]
    val_labels = [0, 1]
    
    # 初始化训练器
    trainer = ClinicalModernBERTTrainer()
    
    # 自定义训练参数
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "clinical_classifier"),
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=3e-5,
        warmup_steps=10,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    
    # 训练
    train_result, eval_metrics = trainer.train_classifier(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        training_args=training_args
    )
    
    if eval_metrics:
        logger.info(f"\n评估结果:")
        for key, value in eval_metrics.items():
            logger.info(f"  {key}: {value:.4f}")
    
    # 预测
    test_texts = ["Patient has chest discomfort and dyspnea."]
    predictions, probs = trainer.predict(test_texts)
    
    logger.info(f"\n预测结果:")
    logger.info(f"  文本: {test_texts[0]}")
    logger.info(f"  预测类别: {predictions[0]}")
    logger.info(f"  概率分布: {probs[0]}")


def example_embedding_extraction():
    """Embedding提取示例"""
    logger.info("="*60)
    logger.info("Token Embedding提取示例")
    logger.info("="*60)
    
    # 初始化提取器
    extractor = ClinicalEmbeddingExtractor()
    
    # 示例文本
    texts = [
        "Hypertension and diabetes mellitus type 2.",
        "Acute myocardial infarction with ST elevation.",
        "Patient has chronic kidney disease stage 3."
    ]
    
    # 1. 批量提取句子embeddings
    logger.info("\n1. 句子级Embeddings (mean pooling):")
    embeddings = extractor.extract_embeddings(
        texts,
        pooling_strategy='mean',
        batch_size=2
    )
    logger.info(f"   Shape: {embeddings.shape}")
    
    # 2. 提取CLS token embeddings
    logger.info("\n2. CLS Token Embeddings:")
    cls_embeddings = extractor.extract_embeddings(
        texts,
        pooling_strategy='cls'
    )
    logger.info(f"   Shape: {cls_embeddings.shape}")
    
    # 3. 获取token级embeddings
    logger.info("\n3. Token级Embeddings:")
    token_result = extractor.get_token_embeddings(texts[0])
    logger.info(f"   文本: {texts[0]}")
    logger.info(f"   Tokens数量: {len(token_result['tokens'])}")
    logger.info(f"   Embeddings shape: {token_result['shape']}")
    logger.info(f"   前5个tokens: {token_result['tokens'][:5]}")


if __name__ == "__main__":
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 运行示例
    example_trainer_classification()
    print("\n" + "="*60 + "\n")
    example_embedding_extraction()