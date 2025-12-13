# gs_lora_utils.py
# 梯度谱 LoRA（GS-LoRA）的辅助函数：
# 1）在原始 RoBERTa 上采样梯度并做 SVD；
# 2）用梯度的奇异向量初始化 LoRA 的 lora_A / lora_B。

from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase, DataCollatorForLanguageModeling

from peft.tuners.lora import LoraLayer


@torch.no_grad()
def _get_target_weight_names(model: PreTrainedModel) -> Dict[str, torch.nn.Parameter]:
    """
    找到所有需要做 LoRA 的线性层权重名字（Q/K/V），返回 name -> param 的映射。

    注意：这里是针对 Roberta self-attention 的命名：
      roberta.encoder.layer.{i}.attention.self.{query,key,value}.weight
    """
    name_to_param = dict(model.named_parameters())
    target = {}

    for name, param in name_to_param.items():
        if not name.endswith(".weight"):
            continue
        if ".attention.self.query.weight" in name:
            target[name] = param
        elif ".attention.self.key.weight" in name:
            target[name] = param
        elif ".attention.self.value.weight" in name:
            target[name] = param

    return target


def compute_gradient_svd_bases(
    model: PreTrainedModel,
    train_dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int = 512,
    num_steps: int = 50,
    batch_size: int = 8,
    device: str = "cuda",
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    在原始 RoBERTa 上，用少量 batch 估计每一层 Q/K/V 权重的梯度矩阵 G，
    然后对 G 做截断 SVD，得到：
        G ≈ U_r Σ_r V_r^T
    返回一个字典：{ weight_name -> {"U": U_r, "Vh": V_r^T} }

    参数：
      - model: 原始 RobertaForMaskedLM（未加 LoRA）
      - train_dataset: 已经过 tokenize 的 train split（Dataset）
      - tokenizer: tokenizer，用于构造 DataCollatorForLanguageModeling
      - max_seq_length: 保持与训练时一致
      - num_steps: 用多少个小 batch 估计梯度（越大越稳定，但时间越长）
      - batch_size: 梯度估计阶段的 batch 大小
      - device: "cuda" 或 "cpu"
    """
    print(f"[GS-LoRA] Start gradient sampling: num_steps={num_steps}, batch_size={batch_size}")
    model.to(device)
    model.train()

    # 只对注意力的 Q/K/V 权重做梯度累积
    target_name_to_param = _get_target_weight_names(model)
    if not target_name_to_param:
        raise RuntimeError("[GS-LoRA] No target attention weights (query/key/value) found in model.")

    grad_acc: Dict[str, torch.Tensor] = {
        name: torch.zeros_like(param.data, device=device)
        for name, param in target_name_to_param.items()
    }

    # 使用 MLM 的 collator（和训练时一致）
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # 这里 train_dataset 已经是 tokenized 数据集，直接 DataLoader
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    step = 0
    for batch in loader:
        if step >= num_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # 累积每个目标权重的梯度
        for name, param in target_name_to_param.items():
            if param.grad is not None:
                grad_acc[name] += param.grad.detach()

        step += 1
        if step % 10 == 0:
            print(f"[GS-LoRA] Collected gradients for {step} steps")

    # 计算 SVD，提取前 r 个奇异向量
    svd_bases: Dict[str, Dict[str, torch.Tensor]] = {}
    print("[GS-LoRA] Computing SVD for each target weight matrix...")
    for name, G in grad_acc.items():
        # 移到 CPU 做 SVD（矩阵不大，比如 768x768）
        G_cpu = G.detach().cpu()
        # full_matrices=False，可以得到 (m, k), (k,), (k, n)
        U, S, Vh = torch.linalg.svd(G_cpu, full_matrices=False)
        svd_bases[name] = {
            "U": U,   # (out_features, k)
            "Vh": Vh, # (k, in_features)
        }

    print(f"[GS-LoRA] SVD bases computed for {len(svd_bases)} matrices.")
    return svd_bases


def init_lora_with_gradient_svd(
    lora_model,
    svd_bases: Dict[str, Dict[str, torch.Tensor]],
    adapter_name: str = "default",
    rank: int = 8,
    init_scale: float = 0.1,
):
    """
    用梯度 SVD 的前 r 个奇异向量初始化 LoRA：

      - 对每个加了 LoRA 的 Linear（query/key/value）：
        ΔW_lora ≈ (init_scale^2) * U_r V_r^T
        其中 U_r / V_r^T 来自梯度矩阵的 SVD。

    这里保持结构与 LoRA 完全相同，只改初始化方向：
      lora_A.weight ← init_scale * V_r^T
      lora_B.weight ← init_scale * U_r
    """
    print(f"[GS-LoRA] Initializing LoRA weights from gradient SVD (rank={rank}, init_scale={init_scale})")

    # 重要：LoRAModel 里真正挂 LoRA 的模块在 base_model 下面
    base_model = getattr(lora_model, "base_model", None)
    if base_model is None:
        raise RuntimeError("[GS-LoRA] lora_model.base_model is None. Are you sure this is a LoraModel?")

    num_inited = 0

    for module_name, module in base_model.named_modules():
        # 只处理带 LoRA 的 Linear（LoraLayer）
        if not isinstance(module, LoraLayer):
            continue
        if adapter_name not in module.lora_A or adapter_name not in module.lora_B:
            continue

        weight_name = module_name + ".weight"  # 与原始 model.named_parameters() 中的名字对齐
        if weight_name not in svd_bases:
            continue

        U = svd_bases[weight_name]["U"]   # (out_features, k)
        Vh = svd_bases[weight_name]["Vh"] # (k, in_features)

        lora_A = module.lora_A[adapter_name]  # nn.Linear(in_features, r)
        lora_B = module.lora_B[adapter_name]  # nn.Linear(r, out_features)

        out_features, in_features = module.base_layer.weight.shape
        r = rank

        # 截取前 r 个奇异向量，并按 LoRA 的形状裁剪
        U_r = U[:, :r]              # (out_features, r)
        Vh_r = Vh[:r, :]            # (r, in_features)

        # 与 layer 的尺寸对齐（以防未来有维度变化）
        U_r = U_r[:out_features, :r].contiguous()
        Vh_r = Vh_r[:r, :in_features].contiguous()

        # 拷贝到 LoRA 权重，带一个缩放系数避免扰动过大
        with torch.no_grad():
            lora_A.weight.copy_(init_scale * Vh_r.to(lora_A.weight.device))
            lora_B.weight.copy_(init_scale * U_r.to(lora_B.weight.device))

        num_inited += 1

    print(f"[GS-LoRA] Initialized {num_inited} LoRA layers with gradient SVD bases.")
