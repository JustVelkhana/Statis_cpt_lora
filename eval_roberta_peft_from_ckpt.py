import argparse
import json
import math
import os
import random
import time

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
)
from peft import LoraModel

from peft import LoraModel


# -------------------------------
# 工具函数：随机种子
# -------------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------
# 数据集加载（PiSSA metamath/python）
# -------------------------------
def load_pissa_dataset(sub_task: str, hf_base_url: str):
    """
    从 PiSSA 数据集镜像加载 JSON 数据。
      - {hf_base_url}/metamath/train.json
      - {hf_base_url}/metamath/test.json
      - {hf_base_url}/python/train.json
      - {hf_base_url}/python/test.json
    """
    assert sub_task in ["metamath", "python"]

    train_url = f"{hf_base_url}/{sub_task}/train.json"
    val_url = f"{hf_base_url}/{sub_task}/test.json"  # 用 test 作为 validation

    data_files = {"train": train_url, "validation": val_url}
    print(f"[Data] Use data_files = {data_files}")

    raw_datasets = load_dataset("json", data_files=data_files)
    return raw_datasets


# -------------------------------
# 把一条样本拼成一个长文本（用于 MLM）
# -------------------------------
def build_text(instruction, input_text, output_text):
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    output_text = (output_text or "").strip()

    if input_text:
        text = (
            "Instruction:\n"
            + instruction
            + "\n\nInput:\n"
            + input_text
            + "\n\nOutput:\n"
            + output_text
        )
    else:
        text = "Instruction:\n" + instruction + "\n\nOutput:\n" + output_text
    return text


# -------------------------------
# Tokenize 数据集（和训练脚本保持一致）
# -------------------------------
def tokenize_dataset(raw_datasets, tokenizer, max_seq_length: int):
    def preprocess_fn(batch):
        texts = [
            build_text(ins, inp, out)
            for ins, inp, out in zip(
                batch["instruction"], batch["input"], batch["output"]
            )
        ]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        return tokenized

    column_names = list(raw_datasets["train"].column_names)
    tokenized_datasets = raw_datasets.map(
        preprocess_fn,
        batched=True,
        remove_columns=column_names,
        desc="Tokenizing dataset",
    )
    return tokenized_datasets


# -------------------------------
# 统计参数量
# -------------------------------
def count_parameters(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


# -------------------------------
# 根据方法与 checkpoint 构建模型
# -------------------------------
from peft import PeftModel  # 确保顶部已经有这个 import

def build_model_from_ckpt(method: str,
                          ckpt_dir: str,
                          model_name: str,
                          lora_r: int,
                          lora_alpha: int,
                          lora_dropout: float,
                          lora_target_modules: str,
                          device: str):
    if method == "full":
        # 对 full：checkpoint-xxxx 里已经是一个 RobertaForMaskedLM
        print(f"[Model] Loading FULL model from {ckpt_dir}")
        model = RobertaForMaskedLM.from_pretrained(ckpt_dir)
        # 全量参数都视为可训练（用于统计）
        for p in model.parameters():
            p.requires_grad = True
    else:
        # 对 LoRA / GS-LoRA：先加载 base model，再从 checkpoint 目录加载 LoRA adapter
        print(f"[Model] Loading base model {model_name}")
        base_model = RobertaForMaskedLM.from_pretrained(model_name)

        print(f"[Model] Loading LoRA adapter from {ckpt_dir}")
        # 使用 PeftModel.from_pretrained 来加载 LoRA 权重
        model = PeftModel.from_pretrained(
            base_model,
            ckpt_dir,
            adapter_name="default",
        )

        # 只把 LoRA 参数标记为 requires_grad=True，方便后面统计 trainable_params
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if method == "lora":
            print("[Model] Use standard LoRA for evaluation.")
        elif method == "gs_lora":
            print("[Model] Use GS-LoRA for evaluation (结构与 LoRA 相同).")

    model.to(device)
    return model


# -------------------------------
# 主评估流程（纯手写循环，不用 Trainer）
# -------------------------------
def evaluate_model(model,
                   eval_dataset,
                   tokenizer,
                   batch_size: int,
                   device: str):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )
    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    model.eval()
    n_steps = 0
    n_samples = 0
    sum_loss = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss  # 平均到 batch 里的 tokens 上
            bs = batch["input_ids"].size(0)

            sum_loss += loss.item() * bs
            n_samples += bs
            n_steps += 1

    eval_runtime = time.time() - t0
    eval_loss = sum_loss / max(n_samples, 1)

    eval_samples_per_second = n_samples / eval_runtime if eval_runtime > 0 else 0.0
    eval_steps_per_second = n_steps / eval_runtime if eval_runtime > 0 else 0.0

    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    else:
        peak_mem_mb = 0.0

    try:
        mlm_ppl = math.exp(eval_loss)
    except OverflowError:
        mlm_ppl = float("inf")

    metrics = {
        "eval_loss": float(eval_loss),
        "mlm_loss": float(eval_loss),
        "mlm_perplexity": float(mlm_ppl),
        "eval_runtime": float(eval_runtime),
        "eval_samples_per_second": float(eval_samples_per_second),
        "eval_steps_per_second": float(eval_steps_per_second),
        "peak_gpu_mem_MB": float(peak_mem_mb),
    }

    return metrics


# -------------------------------
# 参数解析
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str,
                        choices=["full", "lora", "gs_lora"],
                        required=True,
                        help="full / lora / gs_lora")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="checkpoint 目录，例如 outputs/.../checkpoint-24688")
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--sub_task", type=str,
                        choices=["metamath", "python"],
                        default="metamath")
    parser.add_argument(
        "--hf_base_url",
        type=str,
        default="https://hf-mirror.com/datasets/fxmeng/pissa-dataset/resolve/main",
    )
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # LoRA 结构信息（用于从 checkpoint 恢复模型时数结构）
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_target_modules", type=str,
                        default="query,key,value")

    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="评估结果保存路径（默认写到 ckpt_dir/eval_results_from_ckpt.json）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("[Config]", args)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载数据 + tokenizer
    raw_datasets = load_pissa_dataset(args.sub_task, args.hf_base_url)
    print(f"[Data] Raw columns: {raw_datasets['train'].column_names}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.cls_token is not None:
            tokenizer.pad_token = tokenizer.cls_token

    tokenized_datasets = tokenize_dataset(
        raw_datasets, tokenizer, args.max_seq_length
    )
    eval_dataset = tokenized_datasets["validation"]
    print(f"[Data] eval len = {len(eval_dataset)}")

    # 2. 从 checkpoint 构建模型
    model = build_model_from_ckpt(
        method=args.method,
        ckpt_dir=args.ckpt_dir,
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        device=device,
    )

    # 3. 统计参数量
    total_params, trainable_params = count_parameters(model)
    trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0
    print(
        f"[Model] total_params = {total_params:,}, "
        f"trainable_params = {trainable_params:,}, "
        f"ratio = {trainable_ratio * 100:.4f}%"
    )

    # 4. 评估
    print("[Eval] Evaluating from checkpoint...")
    eval_metrics = evaluate_model(
        model=model,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        batch_size=args.per_device_eval_batch_size,
        device=device,
    )

    # 补充参数统计信息
    eval_metrics["total_params"] = float(total_params)
    eval_metrics["trainable_params"] = float(trainable_params)
    eval_metrics["trainable_params_ratio"] = float(trainable_ratio)
    eval_metrics["epoch"] = 1.0  # 你当前都是 1 epoch，可以按需修改

    print("***** Eval metrics from checkpoint *****")
    for k, v in eval_metrics.items():
        print(f"{k}: {v}")

    # 5. 保存到 json
    if args.output_json is None:
        save_path = os.path.join(args.ckpt_dir, "eval_results_from_ckpt.json")
    else:
        save_path = args.output_json

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"[Save] Eval metrics saved to {save_path}")


if __name__ == "__main__":
    main()
