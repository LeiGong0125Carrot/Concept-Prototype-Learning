#!/bin/bash

# LoRA 概念原型学习训练脚本 (后台运行版本)

set -e  # 遇到错误立即退出

# 切换到项目目录
cd /bigtemp/nkw3mr/concept-prototype-learing

# 设置环境变量
export TRITON_CACHE_DIR="/bigtemp/nkw3mr/triton_cache"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 使用随机端口避免冲突
export MASTER_PORT=$((29500 + RANDOM % 1000))

# 创建日志目录
mkdir -p logs

# 生成日志文件名
LOG_FILE="logs/training_$(date +%Y%m%d_%H%M%S).log"

# 显示配置信息
echo "开始LoRA概念原型学习训练..."
echo "时间: $(date)"
echo "工作目录: $(pwd)"
echo "GPU配置: 0,1 (DeepSpeed)"
echo "Master端口: $MASTER_PORT"
echo "日志文件: $LOG_FILE"
echo "后台运行模式 - 即使SSH断开也会继续训练"

# 使用nohup在后台运行训练
nohup bash -c "
    CUDA_VISIBLE_DEVICES=0,1 deepspeed --num_gpus=2 train_peft_deepspeed.py \
        --data_dir /bigtemp/nkw3mr/cnlp_test/long-clinical-doc/datasets/30 \
        --num_epochs 20 \
        --batch_size 7 \
        --learning_rate 5e-5 \
        --max_length 8192 \
        --deepspeed /bigtemp/nkw3mr/concept-prototype-learing/deepspeed_config_mixed_precision.json \
        --report_to wandb \
        --do_train \
        --do_eval \
        --do_predict \
        --use_class_weights \
        --merge_and_save_peft \
        --use_peft \
        --lora_r 8 \
        --lora_alpha 32 \
        --lora_dropout 0.20 \
        --lora_target_modules query value key dense
" > "$LOG_FILE" 2>&1 &

# 获取进程ID
TRAIN_PID=$!

echo ""
echo "✅ 训练已在后台启动!"
echo "进程ID: $TRAIN_PID"
echo ""
echo "监控命令:"
echo "  查看日志: tail -f $LOG_FILE"
echo "  查看进程: ps aux | grep $TRAIN_PID"
echo "  停止训练: kill $TRAIN_PID"
echo ""
echo "训练将在后台继续运行，即使你断开SSH连接"

# 等待几秒钟检查进程是否正常启动
sleep 5
if ps -p $TRAIN_PID > /dev/null 2>&1; then
    echo "✅ 训练进程正在运行中..."
else
    echo "❌ 训练进程启动失败，请检查日志文件: $LOG_FILE"
fi