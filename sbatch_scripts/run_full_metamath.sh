#!/bin/bash
#SBATCH --job-name=full_mm          # 任务名称
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                # 申请 1 张 GPU
#SBATCH --cpus-per-task=4           # CPU 核数
#SBATCH --mem=32G                   # 内存
#SBATCH --time=12:00:00             # 最长运行时间
#SBATCH --output=logs/full_mm_%j.out  # 标准输出

# 1) 加载环境（请根据你集群的模块系统调整）
# module load anaconda/3
# module load cuda/11.8
# module load gcc/11.2

# # 2) 激活 conda 环境
# source ~/.bashrc
# conda activate roberta_peft

# 3) 可选：设置 HuggingFace 镜像，避免下载过慢
export HF_ENDPOINT="https://hf-mirror.com"

# 4) 切换到项目目录（改成你自己的路径）
cd /home/liang/Statis_cpt_lora

# 5) 启动训练（全参数微调 metamath）
python train_roberta_peft.py \
  --sub_task metamath \
  --method full \
  --output_dir outputs/roberta_full_metamath \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --max_seq_length 512 \
  --logging_steps 100 \
  --eval_steps 500 \
  --save_steps 1000 \
  --fp16
