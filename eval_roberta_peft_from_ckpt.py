import os
import json
import math
from typing import List, Dict

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    set_seed,
)

try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# ------------------ 参数解析 ------------------

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate RoBERTa / RoBERTa+LoRA checkpoint on PiSSA metamath/python"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="checkpoint 路径，例如 outputs/roberta_full_metamath/checkpoint-5000",
    )
    parser.add_argument(
        "--sub_task",
        type=str,
        choices=["metamath", "python"],
        required=True,
        help="使用 metamath 或 python 子数据集",
    )
    parser.add_argument(
        "--hf_base_url",
        type=str,
        default="https://hf-mirror.com/datasets/fxmeng/pissa-dataset/resolve/main",
        help="PiSSA 数据根 URL",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="最大序列长度",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="每个设备的 eval batch size",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="用于调试，限制验证集样本数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()
    return args


# ------------------ 数据加载 & 预处理（和训练脚本一致） ------------------

def build_pissa_data_urls(args) -> Dict[str, str]:
    base = args.hf_base_url.rstrip("/")
    if args.sub_task == "metamath":
        train_url = f"{base}/metamath/train.json"
        test_url = f"{base}/metamath/test.json"
    elif args.sub_task == "python":
        train_url = f"{base}/python/train.json"
        test_url = f"{base}/python/test.json"
    else:
        raise ValueError(f"Unknown sub_task: {args.sub_task}")

    data_files = {"train": train_url, "validation": test_url}
    print(f"[Data] Use data_files = {data_files}")
    return data_files


def load_and_tokenize_dataset(args, tokenizer):
    data_files = build_pissa_data_urls(args)
    raw_datasets = load_dataset("json", data_files=data_files)
    column_names = raw_datasets["train"].column_names
    print(f"[Data] Raw columns: {column_names}")

    def preprocess_function(batch):
        # batch-wise 处理 instruction / input / output
        any_key = next(iter(batch.keys()))
        batch_size = len(batch[any_key])
        texts: List[str] = []

        for i in range(batch_size):
            instr = batch["instruction"][i] if "instruction" in batch else ""
            inp = batch["input"][i] if "input" in batch else ""
            out = batch["output"][i] if "output" in batch else ""

            text_parts = []
            if isinstance(instr, str) and instr.strip():
                text_parts.append("Instruction: " + instr.strip())
            if isinstance(inp, str) and inp.strip():
                text_parts.append("Input: " + inp.strip())
            if isinstance(out, str) and out.strip():
                text_parts.append("Output: " + out.strip())

            if not text_parts:
                if "text" in batch:
                    text = batch["text"][i]
                else:
                    raise ValueError("Example has no usable instruction/input/output/text fields")
            else:
                text = "\n".join(text_parts)

            texts.append(text)

        tokenized = tokenizer(
            texts,
            max_length=args.max_seq_length,
            truncation=True,
            padding="max_length",
        )
        return tokenized

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Tokenizing dataset",
    )

    if args.max_eval_samples is not None:
        tokenized_datasets["validation"] = tokenized_datasets["validation"].select(
            range(args.max_eval_samples)
        )

    print(f"[Data] tokenized val len = {len(tokenized_datasets['validation'])}")
    return tokenized_datasets


# ------------------ 加载模型（自动识别 full / LoRA） ------------------

def load_model_and_tokenizer_from_checkpoint(checkpoint_dir: str):
    """
    - 如果 checkpoint_dir 里有 adapter_config.json，则认为是 LoRA/PEFT 模型：
        * 使用 PeftConfig 读取 base_model_name_or_path
        * 再用 PeftModel.from_pretrained 加载
    - 否则认为是普通 RoBERTa MLM checkpoint：
        * 直接 RobertaForMaskedLM.from_pretrained(checkpoint_dir)
    """
    checkpoint_dir = checkpoint_dir.rstrip("/")
    print(f"[Model] Loading from checkpoint_dir = {checkpoint_dir}")

    # tokenizer 直接从 checkpoint 加载，里面有 vocab / merges 等
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    adapter_cfg_path = os.path.join(checkpoint_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg_path):
        if not PEFT_AVAILABLE:
            raise RuntimeError(
                f"Detected adapter_config.json in {checkpoint_dir}, "
                "but peft is not installed. Please pip install peft."
            )
        print("[Model] Detected LoRA/PEFT checkpoint (adapter_config.json found).")
        peft_config = PeftConfig.from_pretrained(checkpoint_dir)
        base_model_name = peft_config.base_model_name_or_path or "roberta-base"
        print(f"[Model] Base model = {base_model_name}")

        base_model = RobertaForMaskedLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    else:
        print("[Model] No adapter_config.json, treating as full RoBERTa MLM checkpoint.")
        model = RobertaForMaskedLM.from_pretrained(checkpoint_dir)

    return model, tokenizer


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ------------------ 构造 TrainingArguments（只用于 eval） ------------------

def build_eval_training_args(output_root: str, args) -> TrainingArguments:
    os.makedirs(output_root, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_root,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        # 以下都是老版本 transformers 会支持的基础参数
        do_train=False,   # 某些版本没有这个参数的话可以删掉
        do_eval=True,
        seed=args.seed,
        logging_steps=100,
    )
    return training_args


# ------------------ 主流程：evaluate 并存 metrics ------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    checkpoint_dir = args.checkpoint_dir.rstrip("/")
    # root 输出目录：写 eval_results.json 在这里
    output_root = os.path.dirname(checkpoint_dir)
    print(f"[Main] checkpoint_dir = {checkpoint_dir}")
    print(f"[Main] output_root    = {output_root}")

    # 1) 加载模型 & tokenizer
    model, tokenizer = load_model_and_tokenizer_from_checkpoint(checkpoint_dir)

    # 2) 加载 & tokenize 数据集
    tokenized_datasets = load_and_tokenize_dataset(args, tokenizer)

    # 3) collator：MLM 随机 mask
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # 4) TrainingArguments（只用来控制 eval dataloader）
    training_args = build_eval_training_args(output_root, args)

    # 5) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 6) 显存统计
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("[Eval] Start evaluation on validation set...")
    metrics = trainer.evaluate()

    # 7) 追加我们需要的字段：mlm_loss, mlm_perplexity, 参数量, 显存峰值等
    eval_loss = float(metrics.get("eval_loss", float("nan")))
    if not math.isnan(eval_loss):
        mlm_loss = eval_loss
        mlm_ppl = math.exp(mlm_loss)
    else:
        mlm_loss = float("nan")
        mlm_ppl = float("nan")

    metrics["mlm_loss"] = mlm_loss
    metrics["mlm_perplexity"] = mlm_ppl

    total_params, trainable_params = count_parameters(model)
    metrics["total_params"] = int(total_params)
    metrics["trainable_params"] = int(trainable_params)
    metrics["trainable_params_ratio"] = float(trainable_params) / float(total_params) if total_params > 0 else float("nan")

    if torch.cuda.is_available():
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        metrics["peak_gpu_mem_MB"] = peak_mem_bytes / (1024 ** 2)

    print("\n***** Eval metrics *****")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    # 8) 保存到 output_root/eval_results.json（供分析脚本使用）
    eval_path = os.path.join(output_root, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n[Save] Eval metrics saved to {eval_path}")


if __name__ == "__main__":
    main()
