"""
Clinical Longformer, BigBird, and Modern BERT模型配置和管理
这三种模型都支持更长的序列处理，适合临床文档分析
"""

import os
from pathlib import Path
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoConfig,
    LongformerModel,
    LongformerTokenizer,
    BigBirdModel,
    BigBirdTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering
)
import torch
import json
from typing import Optional, Dict, Tuple

# 设置项目基础路径
PROJECT_BASE = Path("/bigtemp/nkw3mr/concept-prototype-learing")

# 模型缓存目录配置
MODEL_CACHE_BASE = PROJECT_BASE / "models"
CLINICAL_LONGFORMER_DIR = MODEL_CACHE_BASE / "clinical_longformer"
CLINICAL_BIGBIRD_DIR = MODEL_CACHE_BASE / "clinical_bigbird"  
CLINICAL_MODERN_BERT_DIR = MODEL_CACHE_BASE / "clinical_modern_bert"

# 创建必要的目录
for dir_path in [CLINICAL_LONGFORMER_DIR, CLINICAL_BIGBIRD_DIR, CLINICAL_MODERN_BERT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 设置Hugging Face缓存环境变量
os.environ['HF_HOME'] = str(PROJECT_BASE / "hf_cache")
os.environ['TRANSFORMERS_CACHE'] = str(MODEL_CACHE_BASE)
os.environ['HF_DATASETS_CACHE'] = str(PROJECT_BASE / "datasets_cache")

# 模型配置字典
CLINICAL_MODELS = {
    "clinical_longformer": {
        "model_name": "yikuan8/Clinical-Longformer",
        "cache_dir": CLINICAL_LONGFORMER_DIR,
        "max_length": 4096,
        "model_type": "longformer",
        "description": "Clinical Longformer支持最长4096个token，适合处理长篇临床记录"
    },
    "clinical_bigbird": {
        "model_name": "yikuan8/Clinical-BigBird",  
        "cache_dir": CLINICAL_BIGBIRD_DIR,
        "max_length": 4096,
        "model_type": "bigbird",
        "description": "Clinical BigBird使用稀疏注意力机制，在长文档处理上更高效"
    },
    "clinical_modern_bert": {
        "model_name": "Simonlee711/Clinical_ModernBERT",  # Clinical ModernBERT
        "cache_dir": CLINICAL_MODERN_BERT_DIR,
        "max_length": 8192,  # ModernBERT支持更长的序列
        "model_type": "modernbert",
        "description": "Clinical ModernBERT - 最新的临床文本预训练模型，支持8192 tokens"
    }
}


class ClinicalModelManager:
    """管理Clinical Longformer, BigBird和Modern BERT模型"""
    
    def __init__(self, model_type: str):
        """
        初始化模型管理器
        
        Args:
            model_type: 'clinical_longformer', 'clinical_bigbird', 或 'clinical_modern_bert'
        """
        if model_type not in CLINICAL_MODELS:
            raise ValueError(f"不支持的模型类型: {model_type}. 请选择: {list(CLINICAL_MODELS.keys())}")
        
        self.model_type = model_type
        self.config = CLINICAL_MODELS[model_type]
        self.model = None
        self.tokenizer = None
        
    def download_and_save(self, force_download: bool = False) -> str:
        """
        下载模型并保存到本地缓存目录
        
        Args:
            force_download: 是否强制重新下载
            
        Returns:
            本地模型路径
        """
        cache_dir = self.config["cache_dir"]
        model_name = self.config["model_name"]
        
        # 检查模型是否已存在
        config_file = cache_dir / "config.json"
        if config_file.exists() and not force_download:
            print(f"模型已存在于 {cache_dir}")
            return str(cache_dir)
        
        print(f"正在下载 {self.model_type} 模型...")
        print(f"模型: {model_name}")
        print(f"保存路径: {cache_dir}")
        
        try:
            # 下载并保存tokenizer
            print("下载tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                trust_remote_code=True
            )
            tokenizer.save_pretrained(str(cache_dir))
            
            # 下载并保存model config
            print("下载模型配置...")
            config = AutoConfig.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                trust_remote_code=True
            )
            config.save_pretrained(str(cache_dir))
            
            # 下载并保存model
            print("下载模型权重（这可能需要一些时间）...")
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                trust_remote_code=True
            )
            model.save_pretrained(str(cache_dir))
            
            # 保存下载信息
            info = {
                "model_type": self.model_type,
                "original_model": model_name,
                "max_length": self.config["max_length"],
                "description": self.config["description"],
                "download_time": str(Path.ctime(Path(cache_dir)))
            }
            
            with open(cache_dir / "model_info.json", "w") as f:
                json.dump(info, f, indent=2)
            
            print(f"✓ 模型成功下载并保存到: {cache_dir}")
            return str(cache_dir)
            
        except Exception as e:
            print(f"下载模型时出错: {e}")
            # 尝试使用备选模型
            if self.model_type == "clinical_longformer":
                print("尝试使用备选Longformer模型...")
                return self._download_alternative("longformer_base")
            elif self.model_type == "clinical_bigbird":
                print("尝试使用备选BigBird模型...")
                return self._download_alternative("bigbird_base")
            else:
                raise
    
    def _download_alternative(self, alt_key: str) -> str:
        """下载备选模型"""
        alt_model = ALTERNATIVE_MODELS[alt_key]
        cache_dir = self.config["cache_dir"]
        
        print(f"使用备选模型: {alt_model}")
        
        tokenizer = AutoTokenizer.from_pretrained(alt_model)
        tokenizer.save_pretrained(str(cache_dir))
        
        model = AutoModel.from_pretrained(alt_model)
        model.save_pretrained(str(cache_dir))
        
        print(f"备选模型已保存到: {cache_dir}")
        return str(cache_dir)
    
    def load_from_cache(self) -> Tuple[object, object]:
        """
        从本地缓存加载模型
        
        Returns:
            (model, tokenizer) 元组
        """
        cache_dir = self.config["cache_dir"]
        
        if not (cache_dir / "config.json").exists():
            raise FileNotFoundError(f"模型未找到，请先下载: {cache_dir}")
        
        print(f"从缓存加载 {self.model_type}...")
        
        # 根据模型类型选择合适的加载方式
        if self.config["model_type"] == "longformer":
            self.tokenizer = LongformerTokenizer.from_pretrained(str(cache_dir))
            self.model = LongformerModel.from_pretrained(str(cache_dir))
        elif self.config["model_type"] == "bigbird":
            self.tokenizer = BigBirdTokenizer.from_pretrained(str(cache_dir))
            self.model = BigBirdModel.from_pretrained(str(cache_dir))
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(str(cache_dir))
            self.model = AutoModel.from_pretrained(str(cache_dir))
        
        print(f"✓ 模型加载成功")
        return self.model, self.tokenizer
    
    def load_for_task(self, task: str, num_labels: Optional[int] = None):
        """
        为特定任务加载模型
        
        Args:
            task: 'classification', 'ner', 'qa'
            num_labels: 分类或NER任务的标签数量
            
        Returns:
            (model, tokenizer) 元组
        """
        cache_dir = self.config["cache_dir"]
        
        if not (cache_dir / "config.json").exists():
            raise FileNotFoundError(f"模型未找到，请先下载")
        
        print(f"为{task}任务加载 {self.model_type}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(str(cache_dir))
        
        if task == "classification":
            if num_labels is None:
                raise ValueError("分类任务需要指定num_labels")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(cache_dir),
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
        elif task == "ner":
            if num_labels is None:
                raise ValueError("NER任务需要指定num_labels")
            self.model = AutoModelForTokenClassification.from_pretrained(
                str(cache_dir),
                num_labels=num_labels,
                ignore_mismatched_sizes=True
            )
        elif task == "qa":
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                str(cache_dir),
                ignore_mismatched_sizes=True
            )
        else:
            raise ValueError(f"不支持的任务类型: {task}")
        
        print(f"✓ {task}模型加载成功")
        return self.model, self.tokenizer
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        cache_dir = self.config["cache_dir"]
        info_file = cache_dir / "model_info.json"
        
        if info_file.exists():
            with open(info_file, "r") as f:
                return json.load(f)
        else:
            return self.config
    
    @staticmethod
    def list_cached_models():
        """列出所有已缓存的模型"""
        cached = []
        for model_type, config in CLINICAL_MODELS.items():
            cache_dir = config["cache_dir"]
            if (cache_dir / "config.json").exists():
                size_mb = sum(
                    f.stat().st_size for f in cache_dir.rglob("*") if f.is_file()
                ) / (1024 * 1024)
                
                cached.append({
                    "type": model_type,
                    "path": str(cache_dir),
                    "size_mb": round(size_mb, 2),
                    "max_length": config["max_length"]
                })
        return cached
    
    @staticmethod
    def compare_models():
        """比较三种模型的特性"""
        comparison = []
        for model_type, config in CLINICAL_MODELS.items():
            comparison.append({
                "Model": model_type.replace("_", " ").title(),
                "Max Length": config["max_length"],
                "Architecture": config["model_type"],
                "Description": config["description"]
            })
        return comparison


# 使用示例
if __name__ == "__main__":
    print("="*60)
    print("Clinical Models Manager - 模型管理工具")
    print("="*60)
    
    # 显示模型比较
    print("\n模型特性比较:")
    print("-"*60)
    comparisons = ClinicalModelManager.compare_models()
    for comp in comparisons:
        print(f"\n{comp['Model']}:")
        print(f"  • 最大长度: {comp['Max Length']} tokens")
        print(f"  • 架构: {comp['Architecture']}")
        print(f"  • 说明: {comp['Description']}")
    
    # 显示已缓存的模型
    print("\n" + "="*60)
    print("已缓存的模型:")
    print("-"*60)
    cached = ClinicalModelManager.list_cached_models()
    if cached:
        for model in cached:
            print(f"  • {model['type']}: {model['size_mb']} MB")
    else:
        print("  暂无缓存模型")
    
    # 下载示例（注释以避免自动下载）
    """
    # 下载Clinical Longformer
    manager = ClinicalModelManager("clinical_longformer")
    manager.download_and_save()
    
    # 加载模型进行分类任务
    model, tokenizer = manager.load_for_task("classification", num_labels=2)
    """