import argparse
import math
import os
import random
import json

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, LoraModel


# -------------------------------
# 工具函数：随机种子
# -------------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------------
# 数据集加载（PISSA metamath/python）
# -------------------------------
def load_pissa_dataset(sub_task: str, hf_base_url: str):
    """
    从 PiSSA 数据集镜像加载 JSON 数据。
    之前已经验证可行的路径是：
      - {hf_base_url}/metamath/train.json
      - {hf_base_url}/metamath/test.json
      - {hf_base_url}/python/train.json
      - {hf_base_url}/python/test.json
    """
    assert sub_task in ["metamath", "python"]

    # 和之前 train_roberta_peft.py 完全对齐的写法
    train_url = f"{hf_base_url}/{sub_task}/train.json"
    val_url = f"{hf_base_url}/{sub_task}/test.json"   # 用 test 作为 validation

    data_files = {"train": train_url, "validation": val_url}
    print(f"[Data] Use data_files = {data_files}")

    # ★ 这里一定要用 "json"，而不是 "jsonl"
    raw_datasets = load_dataset("json", data_files=data_files)
    return raw_datasets



# -------------------------------
# 把一条样本拼成一个长文本（用于 MLM）
# -------------------------------
def build_text(instruction, input_text, output_text):
    """
    原始列：
      - instruction
      - input
      - output
      - type

    简单拼接成：
      Instruction:
      ...
      
      Input:
      ...
      
      Output:
      ...
    """
    instruction = instruction or ""
    input_text = input_text or ""
    output_text = output_text or ""

    instruction = instruction.strip()
    input_text = input_text.strip()
    output_text = output_text.strip()

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
# Tokenize 整个数据集
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
            padding="max_length",  # ★ 改成固定长度 padding
        )
        # 不再手动加 labels，让 DataCollatorForLanguageModeling 自动从 input_ids 复制
        return tokenized

    column_names = list(raw_datasets["train"].column_names)
    tokenized_datasets = raw_datasets.map(
        preprocess_fn,
        batched=True,
        remove_columns=column_names,
        desc="Tokenizing dataset",
    )
    return tokenized_datasets

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

def init_pa_lora(model, rank: int, gamma: float = 1e-2):
    """
    对已经注入 LoRA 的模型，按 PA-LoRA 思路进行初始化：
    - 对每个带 lora_A/lora_B 的层做 SVD(W)，取前 r 个奇异向量
    - 构造 A0, B0，使得 ΔW^(0) 位于主奇异子空间
    - 再用 gamma 控制整体扰动大小

    注意：这里只在 CPU 上做 SVD，避免占用 GPU 显存。
    """
    import math

    print(f"[PA-LoRA] Initialize LoRA with SVD (rank={rank}, gamma={gamma})")

    for name, module in model.named_modules():
        # 只处理带 LoRA 结构的层
        if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
            continue

        # 适配器名字：我们在构建 LoraModel 时用的是 'default'
        if "default" in module.lora_A:
            adapter_name = "default"
        else:
            # 兜底（理论上不会走到这里）
            adapter_name = list(module.lora_A.keys())[0]

        lora_A = module.lora_A[adapter_name]  # nn.Linear(in, r)
        lora_B = module.lora_B[adapter_name]  # nn.Linear(r, out)

        # 拿到基础权重 W（out_features, in_features）
        if hasattr(module, "weight") and module.weight is not None:
            W = module.weight.data.detach().float().cpu()
        elif hasattr(module, "base_layer") and hasattr(module.base_layer, "weight"):
            W = module.base_layer.weight.data.detach().float().cpu()
        else:
            print(f"[PA-LoRA] Skip {name}: no weight found.")
            continue

        out_features, in_features = W.shape
        r = min(rank, out_features, in_features)

        # 计算截断 SVD
        try:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)  # U:[out, k], Vh:[k, in]
        except Exception as e:
            print(f"[PA-LoRA] SVD failed on {name}: {e}. Skip.")
            continue

        U_r = U[:, :r]            # [out, r]
        S_r = S[:r]               # [r]
        V_r = Vh[:r, :].T         # [in, r]

        # 构造 A0, B0:
        # A0 = sqrt(gamma) * Σ_r^{1/2} V_r^T  -> [r, in]
        # B0 = sqrt(gamma) * U_r Σ_r^{1/2}    -> [out, r]
        S_sqrt = torch.sqrt(S_r + 1e-8)             # [r]
        sqrt_gamma = math.sqrt(gamma)

        Vt = V_r.T                                   # [r, in]
        # [r, in] = [r,1] * [r,in]（广播）
        A0 = (S_sqrt.unsqueeze(1) * Vt) * sqrt_gamma

        # [out, r] = [out,r] * [1,r]（广播）
        B0 = (U_r * S_sqrt.unsqueeze(0)) * sqrt_gamma

        # 拷贝到 LoRA 模块中
        with torch.no_grad():
            if lora_A.weight.shape != A0.shape:
                print(f"[PA-LoRA] Shape mismatch in A for {name}: "
                      f"lora_A.weight={tuple(lora_A.weight.shape)}, A0={tuple(A0.shape)}")
                continue
            if lora_B.weight.shape != B0.shape:
                print(f"[PA-LoRA] Shape mismatch in B for {name}: "
                      f"lora_B.weight={tuple(lora_B.weight.shape)}, B0={tuple(B0.shape)}")
                continue

            lora_A.weight.data.copy_(A0.to(lora_A.weight.device))
            lora_B.weight.data.copy_(B0.to(lora_B.weight.device))

        print(f"[PA-LoRA] Initialized layer {name} with rank={r}")


# -------------------------------
# 构建 RoBERTa + （可选）LoRA
# -------------------------------
def build_model_and_tokenizer(args):
    print(f"[Model] Loading tokenizer and model from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # 对 RoBERTa，一般自带 pad_token，这里做一下兜底
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.cls_token is not None:
            tokenizer.pad_token = tokenizer.cls_token

    model = RobertaForMaskedLM.from_pretrained(args.model_name)

    if args.method in ["lora", "pa_lora"]:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",")]
        print(f"[LoRA] target_modules = {target_modules}")

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
        )

        # 注入 LoRA 结构
        model = LoraModel(model, lora_config, "default")

        # 如果是 PA-LoRA，则在此基础上做 SVD 初始化
        if args.method == "pa_lora":
            init_pa_lora(model, rank=args.lora_r, gamma=args.pa_gamma)

        # 冻结 base 参数，只训练 LoRA
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        print(f"[Model] {args.method} wrapped.")

    else:
        print("[Model] Full fine-tuning (no LoRA).")

    total_params, trainable_params = count_parameters(model)
    print(
        f"[Model] total_params = {total_params:,}, "
        f"trainable_params = {trainable_params:,}, "
        f"ratio = {trainable_params / total_params * 100:.4f}%"
    )

    return model, tokenizer, total_params, trainable_params


# -------------------------------
# 自定义 Trainer：覆盖 _save，避免 safetensors 报错
# -------------------------------
class MyTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int = None,
    ):
        """
        自定义 loss 计算：
        - 删除 Trainer/Accelerate 加进去的内部字段 num_items_in_batch
        - 再调用模型的 forward
        """
        # 拷贝一份，避免原 dict 被就地修改
        inputs = dict(inputs)

        # 关键：把内部字段 pop 掉
        inputs.pop("num_items_in_batch", None)

        # 直接调用模型（LoraModel -> RobertaForMaskedLM）
        outputs = model(**inputs)

        # 标准取 loss 的写法：兼容 ModelOutput / dict / tuple
        if hasattr(outputs, "loss"):
            loss = outputs.loss
        elif isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        else:
            # transformers 一般把 loss 放在 outputs[0]
            loss = outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir: str = None, state_dict=None):
        """
        覆盖默认保存逻辑：
        - 不使用 safetensors.save_file（会因为共享权重报错）
        - 改为调用 model.save_pretrained(safe_serialization=False)
        """
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        if state_dict is None:
            state_dict = self.model.state_dict()

        self.model.save_pretrained(
            output_dir,
            state_dict=state_dict,
            safe_serialization=False,
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))


# -------------------------------
# 参数解析
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument(
        "--sub_task", type=str, choices=["metamath", "python"], default="metamath"
    )
    parser.add_argument(
        "--hf_base_url",
        type=str,
        default="https://hf-mirror.com/datasets/fxmeng/pissa-dataset/resolve/main",
    )
    # parser.add_argument(
    #     "--method", type=str, choices=["full", "lora"], default="lora"
    # )
    parser.add_argument(
        "--method", type=str, choices=["full", "lora", "pa_lora"], default="lora",
        help="full: 全量微调; lora: 标准 LoRA; pa_lora: 主成分引导的 LoRA"
    )
    # PA-LoRA 额外超参数：控制初始化扰动大小
    parser.add_argument(
        "--pa_gamma",
        type=float,
        default=1e-2,
        help="PA-LoRA 初始化时的缩放因子 gamma，越小初始扰动越小 (默认 1e-2)。",
    )


    # LoRA 超参数
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    # 对 RoBERTa 来说，自注意力里是 self.query/self.key/self.value
    parser.add_argument("--lora_target_modules", type=str, default="query,key,value")

    parser.add_argument(
        "--output_dir", type=str, default="outputs/roberta_lora_metamath"
    )
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    args = parser.parse_args()
    return args


# -------------------------------
# 主流程：训练 + 评估 + 保存结果
# -------------------------------
def main():
    args = parse_args()
    print("[Config]", args)

    set_seed(args.seed)

    # 1. 加载原始数据
    raw_datasets = load_pissa_dataset(args.sub_task, args.hf_base_url)
    print(f"[Data] Raw columns: {raw_datasets['train'].column_names}")

    # 2. 构建模型 + tokenizer（可选 LoRA）
    model, tokenizer, total_params, trainable_params = build_model_and_tokenizer(args)

    # 3. Tokenize
    tokenized_datasets = tokenize_dataset(raw_datasets, tokenizer, args.max_seq_length)
    if args.max_train_samples is not None:
        tokenized_datasets["train"] = tokenized_datasets["train"].select(
            range(args.max_train_samples)
        )
    if args.max_eval_samples is not None:
        tokenized_datasets["validation"] = tokenized_datasets[
            "validation"
        ].select(range(args.max_eval_samples))

    print(
        f"[Data] tokenized train len = {len(tokenized_datasets['train'])}, "
        f"val len = {len(tokenized_datasets['validation'])}"
    )

    # 4. MLM 的 collator（会自动随机 mask）
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # 5. TrainingArguments
    # 注意：这里只用老版本 Transformers 也支持的字段，不再传 evaluation_strategy 等
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=1,
        prediction_loss_only=True,
        fp16=args.fp16,
        seed=args.seed,

        # 关键一行：不要自动根据模型 forward 签名删列
        remove_unused_columns=False,
    )


    trainer = MyTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    # 6. 统计显存峰值
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 7. 训练
    print("[Train] Start training...")
    train_result = trainer.train()
    trainer.save_model(args.output_dir)  # 会调用自定义 _save

    # 8. 评估
    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics.get("eval_loss", None)
    if eval_loss is not None:
        eval_metrics["mlm_loss"] = float(eval_loss)
        eval_metrics["mlm_perplexity"] = float(math.exp(eval_loss))

    # 9. 记录参数量
    eval_metrics["total_params"] = int(total_params)
    eval_metrics["trainable_params"] = int(trainable_params)
    eval_metrics["trainable_params_ratio"] = float(
        trainable_params / total_params
    )

    # 10. 显存峰值（MB）
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        eval_metrics["peak_gpu_mem_MB"] = float(peak_mem)

    print("***** Eval metrics *****")
    for k, v in eval_metrics.items():
        print(f"{k}: {v}")

    # 11. 保存到 eval_results.json（供后续 analyze 脚本读取）
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2, ensure_ascii=False)
    print(f"[Save] Eval metrics saved to {out_path}")


if __name__ == "__main__":
    main()
