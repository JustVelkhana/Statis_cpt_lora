#!/bin/bash

set -e  # 只要有一个命令报错就退出

# 1. 激活环境（根据你自己的路径改一下）
source ~/miniconda3/etc/profile.d/conda.sh
conda activate roberta_peft

cd /home/leo/Statis_cpt_lora

echo "==== Run 1/3: FULL fine-tuning on Metamath ===="
python train_roberta_lora_simple.py \
  --model_name roberta-base \
  --sub_task metamath \
  --method full \
  --output_dir outputs/roberta_full_metamath_simple \
  --max_seq_length 512 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --logging_steps 100 \
  --save_steps 1000 \
  --seed 42 \
  --fp16

echo "==== Run 2/3: LoRA r=8 on Metamath ===="
python train_roberta_lora_simple.py \
  --model_name roberta-base \
  --sub_task metamath \
  --method lora \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --lora_target_modules query,key,value \
  --output_dir outputs/roberta_lora_r8_metamath \
  --max_seq_length 512 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --logging_steps 100 \
  --save_steps 1000 \
  --seed 42 \
  --fp16

echo "==== Run 3/3: GS-LoRA r=8 on Metamath ===="
python train_roberta_lora_simple.py \
  --model_name roberta-base \
  --sub_task metamath \
  --method gs_lora \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --lora_target_modules query,key,value \
  --output_dir outputs/roberta_gslora_r8_metamath \
  --max_seq_length 512 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --logging_steps 100 \
  --save_steps 1000 \
  --seed 42 \
  --fp16 \
  --gs_num_steps 50 \
  --gs_batch_size 8 \
  --gs_init_scale 0.1

echo "==== All 3 runs finished. You can now run: python analyze_roberta_lora_vs_gslora.py ===="
