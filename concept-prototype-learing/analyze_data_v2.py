#!/usr/bin/env python
"""
数据集分析脚本 V2
分析train/dev/test数据集的类别分布和基本统计
适配格式: {"data": [{"text": ..., "out_hospital_mortality_30": 0/1, "id": ...}]}
"""

import json
from pathlib import Path
import numpy as np

def analyze_dataset(data_dir):
    """分析数据集的详细信息"""
    data_dir = Path(data_dir)
    
    print("\n" + "="*60)
    print(f"数据集分析: {data_dir}")
    print("="*60)
    
    all_stats = {}
    
    for file in ['train.json', 'dev.json', 'test.json']:
        file_path = data_dir / file
        if not file_path.exists():
            print(f"\n{file}: 不存在")
            continue
            
        print(f"\n{file}:")
        print("-" * 40)
        
        with open(file_path) as f:
            data = json.load(f)
        
        # 处理嵌套格式 {"data": [...]}
        if isinstance(data, dict) and 'data' in data:
            actual_data = data['data']
            print(f"  数据格式: 嵌套字典 (键: {list(data.keys())})")
            
            if isinstance(actual_data, list) and len(actual_data) > 0:
                sample = actual_data[0]
                print(f"  样本格式: {list(sample.keys())}")
                
                # 提取标签和文本
                # 标签字段可能是 'out_hospital_mortality_30' 或 'label'
                label_field = None
                if 'out_hospital_mortality_30' in sample:
                    label_field = 'out_hospital_mortality_30'
                elif 'label' in sample:
                    label_field = 'label'
                elif 'labels' in sample:
                    label_field = 'labels'
                
                if label_field:
                    labels = [item[label_field] for item in actual_data]
                    texts = [item.get('text', item.get('texts', '')) for item in actual_data]
                    
                    # 统计标签分布
                    total = len(labels)
                    unique_labels = set(labels)
                    
                    print(f"\n  样本统计:")
                    print(f"    总样本数: {total}")
                    print(f"    标签字段: {label_field}")
                    print(f"    标签类型: {sorted(unique_labels)}")
                    
                    # 计算每个类别的数量
                    label_counts = {}
                    for label in unique_labels:
                        count = labels.count(label)
                        percentage = count / total * 100
                        label_counts[label] = count
                        # 对于mortality任务，0=存活，1=死亡
                        if label_field == 'out_hospital_mortality_30':
                            label_name = "存活" if label == 0 else "死亡"
                            print(f"    类别 {label} ({label_name}): {count} ({percentage:.2f}%)")
                        else:
                            print(f"    类别 {label}: {count} ({percentage:.2f}%)")
                    
                    # 计算类别不平衡比例
                    if len(label_counts) == 2:
                        counts = list(label_counts.values())
                        imbalance_ratio = max(counts) / min(counts)
                        majority_class = max(label_counts, key=label_counts.get)
                        minority_class = min(label_counts, key=label_counts.get)
                        print(f"    类别不平衡比: {imbalance_ratio:.2f}:1 (多数类:{majority_class}, 少数类:{minority_class})")
                        
                        # 计算理论上的类别权重
                        total_samples = sum(counts)
                        num_classes = len(counts)
                        weights = {}
                        for label, count in label_counts.items():
                            weight = total_samples / (count * num_classes)
                            weights[label] = weight
                        print(f"\n  理论类别权重:")
                        for label, weight in sorted(weights.items()):
                            if label_field == 'out_hospital_mortality_30':
                                label_name = "存活" if label == 0 else "死亡"
                                print(f"    类别 {label} ({label_name}): {weight:.4f}")
                            else:
                                print(f"    类别 {label}: {weight:.4f}")
                        if len(weights) == 2:
                            weight_ratio = max(weights.values()) / min(weights.values())
                            print(f"    权重比: {weight_ratio:.2f}:1")
                    
                    # 文本长度统计
                    if texts and len(texts) > 0:
                        text_lengths = []
                        for text in texts:
                            if isinstance(text, str):
                                # 粗略估计token数（按空格分割）
                                tokens = text.split()
                                text_lengths.append(len(tokens))
                        
                        if text_lengths:
                            print(f"\n  文本长度统计 (词数):")
                            print(f"    平均长度: {np.mean(text_lengths):.1f} 词")
                            print(f"    中位数: {np.median(text_lengths):.1f} 词")
                            print(f"    最小值: {min(text_lengths)} 词")
                            print(f"    最大值: {max(text_lengths)} 词")
                            print(f"    标准差: {np.std(text_lengths):.1f} 词")
                            print(f"    95分位: {np.percentile(text_lengths, 95):.1f} 词")
                            
                            # 估算token数（通常是词数的1.3-1.5倍）
                            est_token_avg = np.mean(text_lengths) * 1.4
                            est_token_max = max(text_lengths) * 1.4
                            print(f"\n  估算token长度:")
                            print(f"    平均: ~{est_token_avg:.0f} tokens")
                            print(f"    最大: ~{est_token_max:.0f} tokens")
                    
                    all_stats[file] = {
                        'total': total,
                        'label_counts': label_counts,
                        'imbalance_ratio': imbalance_ratio if len(label_counts) == 2 else None,
                        'label_field': label_field
                    }
                else:
                    print(f"  ⚠️  未找到标签字段")
        else:
            print(f"  ⚠️  未知数据格式")
    
    # 总体统计
    print("\n" + "="*60)
    print("总体统计")
    print("="*60)
    
    if all_stats:
        total_samples = sum(stats['total'] for stats in all_stats.values())
        print(f"总样本数: {total_samples}")
        
        # 检查数据集分割比例
        splits_info = []
        for split_name in ['train.json', 'dev.json', 'test.json']:
            if split_name in all_stats:
                ratio = all_stats[split_name]['total'] / total_samples * 100
                splits_info.append(f"{split_name.replace('.json', '')}: {all_stats[split_name]['total']} ({ratio:.1f}%)")
        print(f"数据集分割: {' | '.join(splits_info)}")
        
        # 检查类别分布一致性
        print("\n类别分布一致性:")
        for split in ['train.json', 'dev.json', 'test.json']:
            if split in all_stats and all_stats[split]['label_counts']:
                counts = all_stats[split]['label_counts']
                total = all_stats[split]['total']
                # 按类别0和1的顺序显示
                class_0 = counts.get(0, 0)
                class_1 = counts.get(1, 0)
                print(f"  {split.replace('.json', ''):5s}: 类0={class_0/total*100:5.1f}% | 类1={class_1/total*100:5.1f}% | 比例={class_0/class_1 if class_1>0 else float('inf'):5.1f}:1")
        
        # 计算总体类别分布
        print("\n总体类别分布:")
        total_class_0 = sum(stats['label_counts'].get(0, 0) for stats in all_stats.values())
        total_class_1 = sum(stats['label_counts'].get(1, 0) for stats in all_stats.values())
        if total_class_0 + total_class_1 > 0:
            print(f"  类别0 (存活): {total_class_0} ({total_class_0/(total_class_0+total_class_1)*100:.1f}%)")
            print(f"  类别1 (死亡): {total_class_1} ({total_class_1/(total_class_0+total_class_1)*100:.1f}%)")
            if total_class_1 > 0:
                print(f"  不平衡比: {total_class_0/total_class_1:.1f}:1")
    
    return all_stats


if __name__ == "__main__":
    import sys
    
    # 默认分析三个数据集
    datasets = [
        "/bigtemp/nkw3mr/concept-prototype-learing/data",
        "/bigtemp/nkw3mr/concept-prototype-learing/data_expanded",
        "/bigtemp/nkw3mr/cnlp_test/long-clinical-doc/datasets/30"
    ]
    
    # 如果提供了命令行参数，使用命令行参数
    if len(sys.argv) > 1:
        datasets = sys.argv[1:]
    
    for data_dir in datasets:
        if Path(data_dir).exists():
            analyze_dataset(data_dir)
        else:
            print(f"\n⚠️  目录不存在: {data_dir}")
    
    print("\n分析完成！")