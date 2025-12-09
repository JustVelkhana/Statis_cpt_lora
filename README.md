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
