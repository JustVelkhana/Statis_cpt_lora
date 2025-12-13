## 1. 环境准备

### 1.1 创建虚拟环境（以 conda 为例）

```bash
# 1) 创建环境
conda create -n roberta_peft python=3.10 -y
conda activate roberta_peft

# 2) 安装 PyTorch（根据自己 GPU/CUDA 版本到官网生成命令）
pip install torch torchvision torchaudio --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/cu121


# 3) 安装 HuggingFace 相关库
pip install "transformers" "datasets" "peft" "accelerate" "evaluate" "scikit-learn"
```

> 提示：如果你在国内，可以把 `HF_ENDPOINT` 设为镜像站，例如：
>
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

## 2. 如何运行这三种实验（给你一组命令模板）

假设你在 `Statis_cpt_lora` 根目录下：
```
conda activate roberta_peft

```


### 2.1 全量微调（full）

```bash
python train_roberta_lora_simple.py \
  --model_name roberta-base \
  --sub_task metamath \
  --method full \
  --output_dir outputs/roberta_full_metamath_simple \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --num_train_epochs 1 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.1 \
  --logging_steps 100 \
  --save_steps 1000 \
  --fp16
```

### 2.2 标准 LoRA（random init）

```bash
python train_roberta_lora_simple.py \
  --model_name roberta-base \
  --sub_task metamath \
  --method lora \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --output_dir outputs/roberta_lora_r8_metamath \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --num_train_epochs 1 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.1 \
  --logging_steps 100 \
  --save_steps 1000 \
  --fp16
```

### 2.3 PA-LoRA（主成分引导初始化）

```bash
python train_roberta_lora_simple.py \
  --model_name roberta-base \
  --sub_task metamath \
  --method pa_lora \
  --lora_r 8 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --pa_gamma 1e-2 \
  --output_dir outputs/roberta_pa_lora_r8_metamath \
  --max_seq_length 512 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --num_train_epochs 1 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.1 \
  --logging_steps 100 \
  --save_steps 1000 \
  --fp16
```