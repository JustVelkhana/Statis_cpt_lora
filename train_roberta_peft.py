import os
import math
import argparse
from typing import Dict, List, Tuple

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
# from peft import LoraConfig, get_peft_model, TaskType
from peft import LoraConfig
from peft.tuners.lora import LoraModel


# -----------------------
# 一些工具函数
# -----------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="RoBERTa-base full fine-tuning vs LoRA on PiSSA metamath / python"
    )

    # 基本设置
    parser.add_argument("--model_name", type=str, default="roberta-base",
                        help="预训练模型名称或本地路径（HuggingFace 格式）")
    parser.add_argument("--sub_task", type=str, choices=["metamath", "python"],
                        default="metamath", help="使用 metamath 或 python 子数据集")
    parser.add_argument("--method", type=str, choices=["full", "lora"],
                        default="lora", help="微调方式：full 全参数 或 lora")

    # LoRA 配置
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA 低秩 r")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA 缩放因子 alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="query,key,value",
        help="应用 LoRA 的子模块名称（逗号分隔），RoBERTa 常用 ['query','key','value']:contentReference[oaicite:2]{index=2}",
    )

    # 数据 URL（默认使用 hf-mirror，可以自行换成 huggingface.co）
    parser.add_argument(
        "--hf_base_url",
        type=str,
        default="https://hf-mirror.com/datasets/fxmeng/pissa-dataset/resolve/main",
        help=(
            "PiSSA 数据根 URL，默认是 hf-mirror 镜像。\n"
            "如果你用官方站，可以改成："
            "https://huggingface.co/datasets/fxmeng/pissa-dataset/resolve/main"
        ),
    )

    # 训练相关
    parser.add_argument("--output_dir", type=str, default="./outputs_roberta_peft",
                        help="输出目录（模型和指标）")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="最大序列长度")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true",
                        help="是否启用 FP16（需 GPU 支持）")

    # 调试用：截断训练/验证样本数
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)

    args = parser.parse_args()
    return args


def build_pissa_data_urls(args) -> Dict[str, str]:
    """
    构造 metamath / python 子数据集的 train / test URL
    """
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


def count_parameters(model) -> Tuple[int, int]:
    """
    返回 (总参数量, 可训练参数量)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_and_tokenize_dataset(args, tokenizer):
    """
    1. 从 json 加载 PiSSA 子数据集 (metamath / python)
    2. 将 instruction / input / output 拼接成一个文本
    3. 使用 MLM 方式训练 RoBERTa（把整段文本当成无监督文本）
    """
    data_files = build_pissa_data_urls(args)

    # 使用 datasets 的 json 加载器
    raw_datasets = load_dataset("json", data_files=data_files)
    # 原始列名，一会儿会全部移除
    column_names = raw_datasets["train"].column_names
    print(f"[Data] Raw columns: {column_names}")

    def preprocess_function(batch):
        """
        batch: dict[str, List[Any]]
        我们将每条样本的 (instruction, input, output) 拼成单个字符串：
        Instruction: ...
        Input: ...
        Output: ...
        然后用 tokenizer 编码，交给 MLM collator 做随机 mask。
        """
        # 计算 batch 大小
        any_key = next(iter(batch.keys()))
        batch_size = len(batch[any_key])

        texts: List[str] = []
        for i in range(batch_size):
            instr = batch["instruction"][i] if "instruction" in batch else ""
            inp = batch["input"][i] if "input" in batch else ""
            out = batch["output"][i] if "output" in batch else ""

            text_parts = []
            if instr and isinstance(instr, str) and instr.strip():
                text_parts.append("Instruction: " + instr.strip())
            if inp and isinstance(inp, str) and inp.strip():
                text_parts.append("Input: " + inp.strip())
            if out and isinstance(out, str) and out.strip():
                text_parts.append("Output: " + out.strip())

            if len(text_parts) == 0:
                # 兜底：如果没有 instruction / input / output，就尝试使用 text 字段
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
            padding="max_length",  # 方便直接按 batch 训练，也便于显存对比
        )
        return tokenized

    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Tokenizing dataset",
    )

    if args.max_train_samples is not None:
        tokenized_datasets["train"] = tokenized_datasets["train"].select(
            range(args.max_train_samples)
        )
    if args.max_eval_samples is not None:
        tokenized_datasets["validation"] = tokenized_datasets["validation"].select(
            range(args.max_eval_samples)
        )

    print(
        f"[Data] tokenized train len = {len(tokenized_datasets['train'])}, "
        f"val len = {len(tokenized_datasets['validation'])}"
    )
    return tokenized_datasets


def build_model_and_tokenizer(args):
    """
    加载 tokenizer 和 RoBERTa MLM 模型，并按需要包一层 LoRA
    """
    print(f"[Model] Loading tokenizer and model from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # RoBERTa 默认有 pad_token，如果你换其它模型注意 pad_token 配置
    if tokenizer.pad_token is None:
        # 兜底：优先用 eos_token，否则用 unk_token
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    model = RobertaForMaskedLM.from_pretrained(args.model_name)

    if args.method == "lora":
        target_modules = [m.strip() for m in args.lora_target_modules.split(",")]

        # 不再使用 TaskType，也不走 PeftModel，只用底层 LoraModel
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
        )

        # 用 LoraModel 直接把 LoRA 注入到 RoBERTa 里
        model = LoraModel(model, lora_config, adapter_name="default")


        # 只训练 LoRA 层（底层 peft 已经提供了这个工具函数）
        if hasattr(model, "mark_only_lora_as_trainable"):
            model.mark_only_lora_as_trainable()

        print("[Model] Wrapped with LoraModel. Trainable parameters:")
        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    else:
        print("[Model] Full fine-tuning (no LoRA).")


    total_params, trainable_params = count_parameters(model)
    print(
        f"[Model] total params = {total_params/1e6:.2f} M, "
        f"trainable params = {trainable_params/1e6:.2f} M "
        f"({trainable_params/total_params*100:.4f}%)"
    )

    return model, tokenizer


def build_training_args(args) -> TrainingArguments:
    """
    构造 HuggingFace TrainingArguments（兼容老版本 transformers）
    只使用一些基础参数，避免 evaluation_strategy / warmup_ratio 等新参数
    """
    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        # 老版本里一般已经有 fp16，如果没有，会报错；到时候我们再针对性处理
        fp16=args.fp16 and torch.cuda.is_available(),
        seed=args.seed,
    )

    # 如果你的 transformers 版本比较新，你也可以在这里手动设置一些属性
    # （这些不是 __init__ 里的参数，直接赋值即可；如果不存在就忽略）
    try:
        setattr(training_args, "save_total_limit", 1)
    except Exception:
        pass

    return training_args



def compute_mlm_metrics(eval_pred):
    """
    对 MLM 任务计算 loss / perplexity：
    - loss: 交叉熵，对应 MLM 的平均负对数似然
    - ppl: perplexity = exp(loss)
    """
    logits, labels = eval_pred
    if isinstance(logits, (tuple, list)):
        logits = logits[0]  # Trainer 有时会返回 (logits,) 结构

    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    # labels 中被 mask 的位置是 -100，需要在 loss 中忽略
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    vocab_size = logits.size(-1)

    loss = loss_fct(
        logits.view(-1, vocab_size),
        labels.view(-1),
    )
    perplexity = torch.exp(loss)
    return {
        "mlm_loss": loss.item(),
        "mlm_perplexity": perplexity.item(),
    }


# -----------------------
# 主函数
# -----------------------

def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"[Config] {args}")

    # 1) 构造模型 & tokenizer（含 LoRA 包裹）
    model, tokenizer = build_model_and_tokenizer(args)

    # 2) 加载 & 预处理数据
    tokenized_datasets = load_and_tokenize_dataset(args, tokenizer)

    # 3) DataCollator：MLM 随机 mask
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,  # RoBERTa/BERT 常用 15% mask 比例:contentReference[oaicite:4]{index=4}
    )

    # 4) 训练参数
    training_args = build_training_args(args)

    # 5) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_mlm_metrics,
    )

    # 6) 显存统计：记录训练过程中的峰值显存
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("[Train] Start training...")
    train_result = trainer.train()
    trainer.save_model()  # 保存模型（含 LoRA adapter 或 full 模型）

    train_metrics = train_result.metrics
    train_metrics["train_samples"] = len(tokenized_datasets["train"])
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    print("[Eval] Start evaluation on validation set...")
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(tokenized_datasets["validation"])

    # 统计参数量
    total_params, trainable_params = count_parameters(model)
    eval_metrics["total_params"] = int(total_params)
    eval_metrics["trainable_params"] = int(trainable_params)
    eval_metrics["trainable_params_ratio"] = float(trainable_params) / float(total_params)

    # 统计显存峰值
    if torch.cuda.is_available():
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        eval_metrics["peak_gpu_mem_MB"] = peak_mem_bytes / (1024 ** 2)

    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    print("***** Final eval metrics *****")
    for k, v in eval_metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
