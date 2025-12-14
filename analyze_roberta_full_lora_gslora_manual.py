import json
import math
import os

import matplotlib.pyplot as plt


SUMMARY_PATH = "eval_from_ckpt_summary_manual.json"


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 按 label 排序，方便展示
    data = sorted(data, key=lambda x: x["label"])
    return data


def print_table(results):
    print("\n================== Metamath: Full vs LoRA vs GS-LoRA ==================\n")
    header = (
        "name".ljust(24)
        + "label".ljust(14)
        + "method".ljust(8)
        + "eval_loss".ljust(12)
        + "mlm_ppl".ljust(12)
        + "trainable_M".ljust(14)
        + "ratio%".ljust(10)
    )
    print(header)
    print("-" * len(header))

    # 找 full baseline
    full_trainable = None
    for r in results:
        if r["method"] == "full":
            full_trainable = r["trainable_params"]
            break

    for r in results:
        trainable_M = r["trainable_params"] / 1e6
        if full_trainable is not None:
            ratio_pct = r["trainable_params"] / full_trainable * 100.0
        else:
            ratio_pct = float("nan")

        line = (
            r["name"].ljust(24)
            + r["label"].ljust(14)
            + r["method"].ljust(8)
            + f"{r['eval_loss']:.4f}".ljust(12)
            + f"{r['mlm_perplexity']:.4f}".ljust(12)
            + f"{trainable_M:.3f}".ljust(14)
            + f"{ratio_pct:.3f}".ljust(10)
        )
        print(line)

    print("\n=================================================================\n")


def plot_perplexity(results, save_path="fig_ppl_metamath.png"):
    labels = [r["label"] for r in results]
    ppl = [r["mlm_perplexity"] for r in results]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, ppl)
    plt.ylabel("MLM Perplexity (↓)")
    plt.title("RoBERTa-base on Metamath: Full vs LoRA vs GS-LoRA")

    # 在柱子上标数值
    for bar, v in zip(bars, ppl):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Save] PPL bar plot saved to {save_path}")


def plot_trainable_params(results, save_path="fig_trainable_params_metamath.png"):
    labels = [r["label"] for r in results]
    trainable_M = [r["trainable_params"] / 1e6 for r in results]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, trainable_M)
    plt.yscale("log")  # log 轴，更直观看到 0.4M vs 124M 的差距
    plt.ylabel("Trainable Parameters (Millions, log scale)")
    plt.title("Trainable Parameter Budget")

    for bar, v in zip(bars, trainable_M):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Save] Trainable params plot saved to {save_path}")


def plot_tradeoff(results, save_path="fig_ppl_vs_params_tradeoff.png"):
    """
    参数量 vs PPL 的简单 trade-off 图：
    x: 可训练参数占比（log10）
    y: PPL
    """
    labels = [r["label"] for r in results]

    # 找 full 作为基准
    full_trainable = None
    for r in results:
        if r["method"] == "full":
            full_trainable = r["trainable_params"]
            break

    x_vals = []
    y_vals = []
    for r in results:
        ratio = r["trainable_params"] / full_trainable
        x_vals.append(math.log10(ratio))
        y_vals.append(r["mlm_perplexity"])

    plt.figure(figsize=(6, 4))
    plt.scatter(x_vals, y_vals)

    for x, y, label in zip(x_vals, y_vals, labels):
        plt.text(x, y, label, fontsize=10, ha="right", va="bottom")

    plt.xlabel("log10(Trainable Params Ratio vs Full)")
    plt.ylabel("MLM Perplexity (↓)")
    plt.title("Perplexity vs Trainable Parameter Ratio")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Save] Trade-off plot saved to {save_path}")


def main():
    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError(f"{SUMMARY_PATH} not found.")

    results = load_results(SUMMARY_PATH)
    print_table(results)

    plot_perplexity(results)
    plot_trainable_params(results)
    plot_tradeoff(results)


if __name__ == "__main__":
    main()
