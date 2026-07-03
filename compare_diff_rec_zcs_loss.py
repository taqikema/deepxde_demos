"""对比扩散-反应 PI-DeepONet 中 standard 与 ZCS 的损失收敛曲线。"""

from pathlib import Path
import logging

import matplotlib

# 服务器/WSL 环境通常没有桌面显示器，Agg 可以直接把图保存到文件。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
STANDARD_LOSS_PATH = SCRIPT_DIR / "outputs" / "diff_rec_aligned" / "loss_history_standard.dat"
ZCS_LOSS_PATH = SCRIPT_DIR / "outputs" / "diff_rec_aligned_zcs" / "loss_history_zcs.dat"
OUTPUT_FIG_PATH = SCRIPT_DIR / "outputs" / "diff_rec_loss_comparison.png"

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_loss_history(path):
    """读取 loss_history_*.dat，返回 step、训练总损失和测试总损失。"""

    if not path.exists():
        raise FileNotFoundError(f"找不到损失历史文件: {path}")

    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != 3:
        raise ValueError(f"{path} 应包含 3 列: step train_total_loss test_total_loss")

    return {
        "step": data[:, 0],
        "train": data[:, 1],
        "test": data[:, 2],
    }


def summarize(label, history):
    """打印最终测试损失和最佳测试损失。"""

    best_index = int(np.argmin(history["test"]))
    logger.info(
        "%s: final_test=%.6e, best_test=%.6e at step %.0f",
        label,
        history["test"][-1],
        history["test"][best_index],
        history["step"][best_index],
    )


def plot_comparison(standard, zcs):
    """把 standard/ZCS 的训练与测试总损失画到同一张图中。"""

    OUTPUT_FIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    axes[0].semilogy(standard["step"], np.maximum(standard["train"], 1e-16), "o-", label="standard train")
    axes[0].semilogy(standard["step"], np.maximum(standard["test"], 1e-16), "s-", label="standard test")
    axes[0].set_title("Standard autodiff")

    axes[1].semilogy(zcs["step"], np.maximum(zcs["train"], 1e-16), "o-", label="ZCS train")
    axes[1].semilogy(zcs["step"], np.maximum(zcs["test"], 1e-16), "s-", label="ZCS test")
    axes[1].set_title("ZCS autodiff")

    for ax in axes:
        ax.set_xlabel("Step")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Total loss")
    fig.suptitle("Diffusion-reaction PI-DeepONet loss comparison")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG_PATH, dpi=150)
    plt.close(fig)

    logger.info("对比图已保存到: %s", OUTPUT_FIG_PATH)


def main():
    logger.info("读取 standard 损失历史: %s", STANDARD_LOSS_PATH)
    standard = load_loss_history(STANDARD_LOSS_PATH)
    logger.info("读取 ZCS 损失历史: %s", ZCS_LOSS_PATH)
    zcs = load_loss_history(ZCS_LOSS_PATH)

    summarize("standard", standard)
    summarize("ZCS", zcs)
    plot_comparison(standard, zcs)


if __name__ == "__main__":
    main()
