import json
import os

import matplotlib.pyplot as plt


# ===================== 配置区 =====================

# 每个实验对应的 trainer_state.json 路径（注意改成你实际的路径）
EXPERIMENTS = [
    {
        "name": "full_metamath_simple",
        "label": "Full FT",
        "state_path": "outputs/roberta_full_metamath_simple/trainer_state.json",
    },
    {
        "name": "lora_r8_metamath",
        "label": "LoRA r=8",
        "state_path": "outputs/roberta_lora_r8_metamath/trainer_state.json",
    },
    {
        "name": "gslora_r8_metamath",
        "label": "GS-LoRA r=8",
        "state_path": "outputs/roberta_gslora_r8_metamath/trainer_state.json",
    },
]

# 平滑窗口大小（单位：记录条数），可以视情况调整
SMOOTH_WINDOW = 200


# ===================== 工具函数 =====================

def load_loss_curve_from_trainer_state(path):
    """
    从 trainer_state.json 解析出训练 loss 曲线：
    - 只保留包含 'loss' 且不包含 'eval_' 的日志记录
    - 使用 'step' 作为横轴
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"trainer_state.json not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)

    log_history = state.get("log_history", [])
    steps = []
    losses = []

    for rec in log_history:
        # eval 相关的日志通常形如 {'eval_loss': ..., 'eval_runtime': ...}
        # 我们只取训练日志：包含 'loss' 且不包含 'eval_loss'
        if "loss" in rec and "eval_loss" not in rec:
            step = rec.get("step", None)
            loss = rec["loss"]
            if step is not None:
                steps.append(step)
                losses.append(loss)

    return steps, losses


def moving_average(values, window):
    """
    简单滑动平均，窗口为 window。
    如果数据点少于窗口，直接返回原始数据。
    """
    if len(values) == 0:
        return [], []
    if len(values) <= window:
        return list(range(len(values))), values

    smoothed = []
    smoothed_x = []

    cumsum = [0.0]
    for v in values:
        cumsum.append(cumsum[-1] + v)

    for i in range(window, len(values) + 1):
        avg = (cumsum[i] - cumsum[i - window]) / window
        smoothed.append(avg)
        smoothed_x.append(i - window // 2)  # 平滑后横轴居中一点

    return smoothed_x, smoothed


# ===================== 绘图主逻辑 =====================

def main():
    curves = []

    for exp in EXPERIMENTS:
        name = exp["name"]
        label = exp["label"]
        state_path = exp["state_path"]

        print(f"[Load] {label} from {state_path}")
        try:
            steps, losses = load_loss_curve_from_trainer_state(state_path)
        except FileNotFoundError as e:
            print(f"[WARN] {e} -> skip this experiment")
            continue

        if not steps:
            print(f"[WARN] No training loss found in {state_path}, skip.")
            continue

        # 对 step / loss 按 step 排序，防止乱序
        pairs = sorted(zip(steps, losses), key=lambda x: x[0])
        steps_sorted = [p[0] for p in pairs]
        losses_sorted = [p[1] for p in pairs]

        # 做一份平滑曲线
        smooth_x_idx, smooth_loss = moving_average(losses_sorted, SMOOTH_WINDOW)
        # 把平滑后的横轴 index 映射回真实 step（简单按比例映射）
        if smooth_x_idx:
            # 这里简单线性插值：index -> step
            total = len(steps_sorted)
            smooth_steps = [
                steps_sorted[int(i * (total - 1) / max(total - 1, 1))]
                for i in smooth_x_idx
            ]
        else:
            smooth_steps = []

        curves.append(
            {
                "name": name,
                "label": label,
                "raw_steps": steps_sorted,
                "raw_losses": losses_sorted,
                "smooth_steps": smooth_steps,
                "smooth_losses": smooth_loss,
            }
        )

    if not curves:
        print("[ERROR] No curves loaded, please check paths.")
        return

    # ================= 绘制训练 loss 曲线 =================
    plt.figure(figsize=(8, 5))

    for c in curves:
        # 原始曲线（虚线、透明一点）
        plt.plot(
            c["raw_steps"],
            c["raw_losses"],
            linestyle=":",
            linewidth=1,
            alpha=0.3,
            label=f"{c['label']} (raw)",
        )
        # 平滑曲线（实线）
        if c["smooth_steps"]:
            plt.plot(
                c["smooth_steps"],
                c["smooth_losses"],
                linewidth=2,
                label=f"{c['label']} (smoothed)",
            )

    plt.xlabel("Global Step")
    plt.ylabel("Training Loss")
    plt.title("RoBERTa-base on Metamath: Training Curves (Full vs LoRA vs GS-LoRA)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()

    save_path = "fig_train_loss_metamath.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Save] Training loss curve saved to {save_path}")


if __name__ == "__main__":
    main()
