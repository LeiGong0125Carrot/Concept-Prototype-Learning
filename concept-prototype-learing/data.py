"""
临床文本二分类数据加载模块
使用固定长度padding，适配Flash Attention
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalBinaryDataset(Dataset):
    """临床文本二分类数据集"""
    
    def __init__(
        self, 
        json_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        label_key: str = "out_hospital_mortality_30"
    ):
        """
        初始化数据集
        
        Args:
            json_path: JSON数据文件路径
            tokenizer: Hugging Face tokenizer
            max_length: 最大序列长度
            label_key: 标签字段名
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_key = label_key
        
        # 加载数据
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理不同的JSON格式
        if isinstance(data, dict) and 'data' in data:
            raw_data = data['data']
        else:
            raw_data = data
        
        self.examples = []
        self._prepare_examples(raw_data)
        
        logger.info(f"从 {json_path} 加载 {len(self.examples)} 条数据")
        
        # 统计标签分布
        label_counts = self._get_label_distribution()
        logger.info(f"标签分布: {label_counts}")
    
    def _prepare_examples(self, raw_data: List[Dict]):
        """准备数据样本"""
        for item in raw_data:
            text = item.get('text', '')
            label = item.get(self.label_key)
            
            if text and label is not None:
                self.examples.append({
                    'text': text,
                    'label': int(label),
                    'id': item.get('id', 'unknown')
                })
    
    def _get_label_distribution(self) -> Dict[int, int]:
        """获取标签分布"""
        label_counts = {}
        for example in self.examples:
            label = example['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        return label_counts
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """返回tokenized的数据 - 不做padding，让DataCollator处理"""
        example = self.examples[idx]
        
        # Tokenize文本 - 不做padding，让DataCollator处理
        encoding = self.tokenizer(
            example['text'],
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            # 不设置padding，让DataCollatorWithPadding处理
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': example['label']  # 不需要转tensor，DataCollator会处理
        }


def create_dataloader(
    json_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    shuffle: bool = False,
    max_length: int = 512,
    label_key: str = "out_hospital_mortality_30",
    num_workers: int = 2
) -> DataLoader:
    """
    创建单个数据加载器
    
    Args:
        json_path: JSON数据文件路径
        tokenizer: tokenizer实例
        batch_size: 批次大小
        shuffle: 是否打乱
        max_length: 最大序列长度
        label_key: 标签字段名
        num_workers: 工作进程数
    
    Returns:
        DataLoader实例
    """
    dataset = ClinicalBinaryDataset(
        json_path=json_path,
        tokenizer=tokenizer,
        max_length=max_length,
        label_key=label_key
    )
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def create_dataloaders(
    data_dir: str,
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    label_key: str = "out_hospital_mortality_30",
    num_workers: int = 2
) -> Dict[str, DataLoader]:
    """
    创建训练、验证、测试数据加载器
    
    Args:
        data_dir: 数据目录
        tokenizer: tokenizer实例
        batch_size: 批次大小
        max_length: 最大序列长度
        label_key: 标签字段名
        num_workers: 工作进程数
    
    Returns:
        包含train/dev/test的DataLoader字典
    """
    dataloaders = {}
    data_path = Path(data_dir)
    
    for split in ['dev', 'train', 'test']:
        json_path = data_path / f'{split}.json'
        
        if json_path.exists():
            dataloaders[split] = create_dataloader(
                json_path=str(json_path),
                tokenizer=tokenizer,
                batch_size=batch_size,
                shuffle=(split == 'train'),
                max_length=max_length,
                label_key=label_key,
                num_workers=num_workers
            )
            logger.info(f"加载 {split} 数据集")
        else:
            logger.warning(f"未找到 {split} 数据文件: {json_path}")
    
    return dataloaders