import os
import json
from dataclasses import dataclass
from typing import Optional, List

import math
import matplotlib.pyplot as plt


# -------- 1. 根据你的实验设计手动列出各个实验 --------

@dataclass
class ExperimentConfig:
    name: str          # 用于图例显示和打印
    dataset: str       # "metamath" / "python"
    method: str        # "full" / "lora"
    rank: Optional[int]  # LoRA rank，如果是 full 就用 None
    output_dir: str    # 对应的输出目录路径


# 根据你实际的输出目录调整这里的路径
BASE_DIR = "/home/leo/Statis_cpt_lora/outputs/eval_results.json"

EXPERIMENTS: List[ExperimentConfig] = [
    # metamath
    ExperimentConfig(
        name="metamath_full",
        dataset="metamath",
        method="full",
        rank=None,
        output_dir=os.path.join(BASE_DIR, "roberta_full_metamath"),
    ),
    # ExperimentConfig(
    #     name="metamath_lora_r4",
    #     dataset="metamath",
    #     method="lora",
    #     rank=4,
    #     output_dir=os.path.join(BASE_DIR, "roberta_lora_r4_metamath"),
    # ),
    # ExperimentConfig(
    #     name="metamath_lora_r8",
    #     dataset="metamath",
    #     method="lora",
    #     rank=8,
    #     output_dir=os.path.join(BASE_DIR, "roberta_lora_r8_metamath"),
    # ),
    # ExperimentConfig(
    #     name="metamath_lora_r16",
    #     dataset="metamath",
    #     method="lora",
    #     rank=16,
    #     output_dir=os.path.join(BASE_DIR, "roberta_lora_r16_metamath"),
    # ),

    # # python
    # ExperimentConfig(
    #     name="python_full",
    #     dataset="python",
    #     method="full",
    #     rank=None,
    #     output_dir=os.path.join(BASE_DIR, "roberta_full_python"),
    # ),
    # ExperimentConfig(
    #     name="python_lora_r4",
    #     dataset="python",
    #     method="lora",
    #     rank=4,
    #     output_dir=os.path.join(BASE_DIR, "roberta_lora_r4_python"),
    # ),
    # ExperimentConfig(
    #     name="python_lora_r8",
    #     dataset="python",
    #     method="lora",
    #     rank=8,
    #     output_dir=os.path.join(BASE_DIR, "roberta_lora_r8_python"),
    # ),
    # ExperimentConfig(
    #     name="python_lora_r16",
    #     dataset="python",
    #     method="lora",
    #     rank=16,
    #     output_dir=os.path.join(BASE_DIR, "roberta_lora_r16_python"),
    # ),
]


# -------- 2. 读取 eval_results.json 并整理成一个列表 --------

@dataclass
class ExperimentResult:
    cfg: ExperimentConfig
    mlm_loss: float
    mlm_ppl: float
    total_params: int
    trainable_params: int
    trainable_ratio: float
    peak_mem_mb: float


def load_experiment_result(cfg: ExperimentConfig) -> Optional[ExperimentResult]:
    eval_path = os.path.join(cfg.output_dir, "eval_results.json")
    if not os.path.exists(eval_path):
        print(f"[WARN] eval_results.json not found for {cfg.name} at {eval_path}")
        return None

    with open(eval_path, "r") as f:
        metrics = json.load(f)

    # 兼容：如果没记录我们的额外字段，先做下兜底
    mlm_loss = float(metrics.get("mlm_loss", metrics.get("eval_loss", math.nan)))
    mlm_ppl = float(metrics.get("mlm_perplexity", math.exp(mlm_loss) if not math.isnan(mlm_loss) else math.nan))
    total_params = int(metrics.get("total_params", -1))
    trainable_params = int(metrics.get("trainable_params", -1))
    trainable_ratio = float(metrics.get("trainable_params_ratio", -1.0))
    peak_mem_mb = float(metrics.get("peak_gpu_mem_MB", -1.0))

    return ExperimentResult(
        cfg=cfg,
        mlm_loss=mlm_loss,
        mlm_ppl=mlm_ppl,
        total_params=total_params,
        trainable_params=trainable_params,
        trainable_ratio=trainable_ratio,
        peak_mem_mb=peak_mem_mb,
    )


def collect_all_results() -> List[ExperimentResult]:
    results = []
    for cfg in EXPERIMENTS:
        res = load_experiment_result(cfg)
        if res is not None:
            results.append(res)
    return results


# -------- 3. 打印一张“结果表”到终端（也可存到 CSV） --------

def print_results_table(results: List[ExperimentResult]):
    print("\n================== 汇总结果表 ==================\n")
    header = (
        f"{'name':<20} {'dataset':<10} {'method':<8} {'rank':<6} "
        f"{'mlm_loss':<12} {'mlm_ppl':<12} "
        f"{'trainable_M':<12} {'total_M':<10} "
        f"{'ratio%':<8} {'peak_mem_MB':<12}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        trainable_M = r.trainable_params / 1e6 if r.trainable_params > 0 else float("nan")
        total_M = r.total_params / 1e6 if r.total_params > 0 else float("nan")
        ratio_pct = r.trainable_ratio * 100 if r.trainable_ratio > 0 else float("nan")

        rank_str = str(r.cfg.rank) if r.cfg.rank is not None else "-"
        print(
            f"{r.cfg.name:<20} {r.cfg.dataset:<10} {r.cfg.method:<8} {rank_str:<6} "
            f"{r.mlm_loss:<12.4f} {r.mlm_ppl:<12.4f} "
            f"{trainable_M:<12.3f} {total_M:<10.3f} "
            f"{ratio_pct:<8.3f} {r.peak_mem_mb:<12.1f}"
        )

    print("\n================================================\n")


# -------- 4. 画图：参数量 vs ppl、显存 vs ppl --------

def plot_param_vs_ppl(results: List[ExperimentResult], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    for dataset in ["metamath", "python"]:
        subset = [r for r in results if r.cfg.dataset == dataset]
        if not subset:
            continue

        plt.figure()
        xs = []
        ys = []
        labels = []

        for r in subset:
            # 横轴：可训练参数量（百万）
            if r.trainable_params <= 0:
                continue
            x = r.trainable_params / 1e6
            y = r.mlm_ppl
            xs.append(x)
            ys.append(y)

            if r.cfg.method == "full":
                label = f"{dataset}_full"
            else:
                label = f"{dataset}_lora_r{r.cfg.rank}"
            labels.append(label)

        # 可能有的实验缺 eval，避免空 plot
        if not xs:
            continue

        plt.scatter(xs, ys)
        for x, y, label in zip(xs, ys, labels):
            plt.text(x, y, label, fontsize=8, ha="center", va="bottom")

        plt.xlabel("Trainable parameters (M)")
        plt.ylabel("Validation MLM perplexity")
        plt.title(f"Param count vs perplexity ({dataset})")
        plt.grid(True)

        out_path = os.path.join(save_dir, f"params_vs_ppl_{dataset}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[FIG] Saved {out_path}")


def plot_mem_vs_ppl(results: List[ExperimentResult], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    for dataset in ["metamath", "python"]:
        subset = [r for r in results if r.cfg.dataset == dataset]
        if not subset:
            continue

        plt.figure()
        xs = []
        ys = []
        labels = []

        for r in subset:
            if r.peak_mem_mb <= 0:
                continue
            x = r.peak_mem_mb / 1024.0  # 转成 GB
            y = r.mlm_ppl
            xs.append(x)
            ys.append(y)

            if r.cfg.method == "full":
                label = f"{dataset}_full"
            else:
                label = f"{dataset}_lora_r{r.cfg.rank}"
            labels.append(label)

        if not xs:
            continue

        plt.scatter(xs, ys)
        for x, y, label in zip(xs, ys, labels):
            plt.text(x, y, label, fontsize=8, ha="center", va="bottom")

        plt.xlabel("Peak GPU memory (GB)")
        plt.ylabel("Validation MLM perplexity")
        plt.title(f"Peak memory vs perplexity ({dataset})")
        plt.grid(True)

        out_path = os.path.join(save_dir, f"mem_vs_ppl_{dataset}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[FIG] Saved {out_path}")


def main():
    results = collect_all_results()
    if not results:
        print("No eval_results.json found. Please check output_dir paths in EXPERIMENTS.")
        return

    print_results_table(results)

    save_dir = os.path.join(BASE_DIR, "analysis_figs")
    plot_param_vs_ppl(results, save_dir)
    plot_mem_vs_ppl(results, save_dir)


if __name__ == "__main__":
    main()
