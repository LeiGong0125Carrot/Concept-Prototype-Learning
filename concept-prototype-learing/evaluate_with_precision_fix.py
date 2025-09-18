#!/usr/bin/env python
"""
精度一致性评估脚本
确保评估结果与训练时保持一致
"""

import torch
import logging
from pathlib import Path
from utility import load_best_peft_model_for_eval, evaluate_dataset_simple
from train_peft_deepspeed import load_datasets
import argparse
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_with_precision_control(
    checkpoint_path: str,
    data_dir: str,
    model_name: str = "Simonlee711/Clinical_ModernBERT",
    batch_size: int = 32,
    use_fp32: bool = False
):
    """
    控制精度的评估函数
    
    Args:
        checkpoint_path: 模型checkpoint路径
        data_dir: 数据目录
        model_name: 基础模型名称
        batch_size: 批次大小
        use_fp32: 是否使用FP32进行评估（更高精度）
    """
    
    logger.info(f"加载模型从: {checkpoint_path}")
    logger.info(f"评估精度模式: {'FP32' if use_fp32 else 'BF16'}")
    
    # 加载模型
    model, tokenizer = load_best_peft_model_for_eval(
        checkpoint_path=checkpoint_path,
        model_name=model_name
    )
    
    # 如果使用FP32，转换模型精度
    if use_fp32:
        logger.info("转换模型到FP32精度...")
        model = model.float()
    
    # 设置确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 加载数据集
    logger.info(f"加载数据集从: {data_dir}")
    datasets = load_datasets(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=8192
    )
    
    results = {}
    
    # 验证集评估
    if datasets.get('dev'):
        logger.info("评估验证集...")
        dev_results = evaluate_dataset_simple(
            model=model,
            tokenizer=tokenizer,
            dataset=datasets['dev'],
            batch_size=batch_size
        )
        results['dev'] = dev_results
        
        logger.info("验证集结果:")
        for key, value in dev_results.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.6f}")
    
    # 测试集评估
    if datasets.get('test'):
        logger.info("评估测试集...")
        test_results = evaluate_dataset_simple(
            model=model,
            tokenizer=tokenizer,
            dataset=datasets['test'],
            batch_size=batch_size
        )
        results['test'] = test_results
        
        logger.info("测试集结果:")
        for key, value in test_results.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.6f}")
    
    return results

def compare_precision_modes(checkpoint_path: str, data_dir: str):
    """
    比较BF16和FP32模式下的评估结果
    """
    logger.info("="*60)
    logger.info("比较不同精度模式的评估结果")
    logger.info("="*60)
    
    # BF16评估
    logger.info("\n1. BF16模式评估:")
    bf16_results = evaluate_with_precision_control(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        use_fp32=False,
        batch_size=16
    )
    
    # FP32评估
    logger.info("\n2. FP32模式评估:")
    fp32_results = evaluate_with_precision_control(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        use_fp32=True,
        batch_size=16
    )
    
    # 比较结果
    logger.info("\n" + "="*60)
    logger.info("结果比较:")
    logger.info("="*60)
    
    for split in ['dev', 'test']:
        if split in bf16_results and split in fp32_results:
            logger.info(f"\n{split.upper()}集差异:")
            bf16 = bf16_results[split]
            fp32 = fp32_results[split]
            
            for key in bf16.keys():
                if isinstance(bf16[key], (int, float)) and isinstance(fp32[key], (int, float)):
                    diff = abs(bf16[key] - fp32[key])
                    logger.info(f"  {key}:")
                    logger.info(f"    BF16: {bf16[key]:.6f}")
                    logger.info(f"    FP32: {fp32[key]:.6f}")
                    logger.info(f"    差异: {diff:.6f}")

def main():
    parser = argparse.ArgumentParser(description="精度一致性评估")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="模型checkpoint路径")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="数据目录")
    parser.add_argument("--model_name", type=str, 
                        default="Simonlee711/Clinical_ModernBERT",
                        help="基础模型名称")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="评估批次大小")
    parser.add_argument("--compare_precision", action="store_true",
                        help="比较BF16和FP32精度")
    parser.add_argument("--use_fp32", action="store_true",
                        help="使用FP32进行评估")
    
    args = parser.parse_args()
    
    if args.compare_precision:
        compare_precision_modes(
            checkpoint_path=args.checkpoint_path,
            data_dir=args.data_dir
        )
    else:
        results = evaluate_with_precision_control(
            checkpoint_path=args.checkpoint_path,
            data_dir=args.data_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            use_fp32=args.use_fp32
        )
        
        # 保存结果
        output_file = Path(args.checkpoint_path).parent / "evaluation_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    main()