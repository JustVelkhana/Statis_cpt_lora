#!/bin/bash
#SBATCH --job-name=lora8_mm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/lora8_mm_%j.out

# source ~/.bashrc
# conda activate roberta_peft
export HF_ENDPOINT="https://hf-mirror.com"

cd /home/leo/Statis_cpt_lora


python train_roberta_peft.py \
  --sub_task metamath \
  --method lora \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --output_dir outputs/roberta_lora_r8_metamath \
  --num_train_epochs 1 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --max_seq_length 512 \
  --logging_steps 100 \
  --eval_steps 500 \
  --save_steps 1000 \
  --fp16
