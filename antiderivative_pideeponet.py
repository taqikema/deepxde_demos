"""DeepXDE 官方 PI-DeepONet 示例：反导数算子。

代码来源：
https://raw.githubusercontent.com/lululxvi/deepxde/master/examples/operator/antiderivative_aligned_pideeponet.py
https://raw.githubusercontent.com/lululxvi/deepxde/master/examples/operator/antiderivative_unaligned_pideeponet.py

本地版本做了四点整理：
1. 在同一份脚本中通过 CONFIG["mode"] 切换 aligned / unaligned 数据格式。
2. 在导入 deepxde 前强制指定 PyTorch 后端。
3. 使用 matplotlib 的 Agg 后端保存图片，避免无图形界面环境中 plt.show() 阻塞。
4. 保存训练好的 PyTorch 模型、损失数据、损失曲线和预测对比图。
"""

from pathlib import Path
import copy
import logging
import os

# DeepXDE 必须在 import 前设置后端；这里按需求固定使用 PyTorch。
os.environ["DDE_BACKEND"] = "pytorch"

import matplotlib

# 服务器/WSL 环境通常没有桌面显示器，Agg 可以直接把图保存到文件。
matplotlib.use("Agg")

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "outputs"

# 将影响训练和数据格式的参数集中放在这里，方便继续调参。
CONFIG = {
    # "aligned" 使用 Cartesian product 数据；"unaligned" 使用逐点配对数据。
    "mode": "aligned",
    "num_domain": 20,
    "num_boundary": 2,
    "num_test_pde": 40,
    "length_scale": 0.2,
    "num_eval_points": 50,
    "num_function": 1000,
    "num_test_aligned": 100,
    "num_test_unaligned": 1000,
    "batch_size": 100,
    "hidden_layers": [128, 128, 128],
    "activation": "tanh",
    "kernel_initializer": "Glorot normal",
    "learning_rate": 5e-4,
    "iterations": 40000,
    "display_every": 1000,
    "num_prediction_points": 50,
}

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def make_config(overrides=None):
    """复制默认配置，并应用调用者提供的覆盖项。"""

    config = copy.deepcopy(CONFIG)
    if overrides:
        config.update(overrides)
    validate_config(config)
    return config


def validate_config(config):
    """检查配置中的模式和关键整数参数，避免 DeepXDE 在更深处报晦涩错误。"""

    if config["mode"] not in {"aligned", "unaligned"}:
        raise ValueError('CONFIG["mode"] 只能是 "aligned" 或 "unaligned"。')

    positive_int_keys = [
        "num_domain",
        "num_boundary",
        "num_test_pde",
        "num_eval_points",
        "num_function",
        "iterations",
        "num_prediction_points",
    ]
    for key in positive_int_keys:
        if int(config[key]) <= 0:
            raise ValueError(f'CONFIG["{key}"] 必须是正整数。')


def antiderivative_pde(x, u, v):
    """反导数问题的物理约束：u_x = v。"""

    return dde.grad.jacobian(u, x) - v


def build_pde(config):
    """构造一维时间区间上的 PDE 和零初值条件。"""

    # 这里的 TimeDomain 等价于反导数示例中的一维区间 [0, 1]。
    geom = dde.geometry.TimeDomain(0, 1)

    # 初值条件 u(0)=0。后面还会通过 output transform 做硬约束。
    ic = dde.icbc.IC(geom, lambda _: 0, lambda _, on_initial: on_initial)

    data_pde = dde.data.PDE(
        geom,
        antiderivative_pde,
        ic,
        num_domain=int(config["num_domain"]),
        num_boundary=int(config["num_boundary"]),
        num_test=int(config["num_test_pde"]),
    )
    return geom, data_pde


def build_operator_data(config, data_pde):
    """根据 aligned / unaligned 模式构造算子学习数据。"""

    eval_points = np.linspace(0, 1, num=int(config["num_eval_points"]))[:, None]

    # GRF 生成输入函数 v(x)，length_scale 控制随机函数的平滑程度。
    func_space = dde.data.GRF(length_scale=float(config["length_scale"]))

    if config["mode"] == "aligned":
        # aligned 数据把所有函数样本和所有空间点组成 Cartesian product。
        data = dde.data.PDEOperatorCartesianProd(
            data_pde,
            func_space,
            eval_points,
            int(config["num_function"]),
            num_test=int(config["num_test_aligned"]),
            batch_size=int(config["batch_size"]),
        )
    else:
        # unaligned 数据使用逐点配对的 (函数样本, 空间点) 训练样本。
        data = dde.data.PDEOperator(
            data_pde,
            func_space,
            eval_points,
            int(config["num_function"]),
            num_test=int(config["num_test_unaligned"]),
        )

    return data, eval_points


def build_network(config):
    """根据数据格式构造匹配的 DeepONet 网络。"""

    branch_layers = [int(config["num_eval_points"]), *config["hidden_layers"]]
    trunk_layers = [1, *config["hidden_layers"]]

    if config["mode"] == "aligned":
        net = dde.nn.DeepONetCartesianProd(
            branch_layers,
            trunk_layers,
            config["activation"],
            config["kernel_initializer"],
        )

        def zero_initial_condition(inputs, outputs):
            # aligned 模式下 outputs 形状为 (函数数, 空间点数)，需要把 x 转成行向量。
            return outputs * torch.transpose(inputs[1], 0, 1)
    else:
        net = dde.nn.DeepONet(
            branch_layers,
            trunk_layers,
            config["activation"],
            config["kernel_initializer"],
        )

        def zero_initial_condition(inputs, outputs):
            # unaligned 模式下 outputs 和 x 都是逐点配对形状，可以直接相乘。
            return outputs * inputs[1]

    # 将网络输出写成 x * NN(v, x)，从结构上保证 u(0)=0。
    net.apply_output_transform(zero_initial_condition)
    return net


def save_convergence_history(loss_history, output_dir):
    """保存 DeepXDE 返回的训练历史，并绘制总损失曲线。"""

    steps = np.asarray(loss_history.steps)
    train_total = np.asarray([np.sum(loss) for loss in loss_history.loss_train])
    test_total = np.asarray([np.sum(loss) for loss in loss_history.loss_test])
    data = np.column_stack([steps, train_total, test_total])

    loss_data_path = output_dir / "loss_history.dat"
    np.savetxt(
        loss_data_path,
        data,
        header="step train_total_loss test_total_loss",
        comments="",
    )

    loss_fig_path = output_dir / "loss_history.png"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(steps, np.maximum(train_total, 1e-16), "o-", label="Train loss")
    ax.semilogy(steps, np.maximum(test_total, 1e-16), "s-", label="Test loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Antiderivative PI-DeepONet convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_fig_path, dpi=150)
    plt.close(fig)

    logger.info("收敛数据已保存到: %s", loss_data_path)
    logger.info("收敛曲线已保存到: %s", loss_fig_path)


def analytical_solution(x):
    """v(x)=sin(pi*x) 时，u(x)=int_0^x v(s) ds=(1-cos(pi*x))/pi。"""

    return (1 - np.cos(np.pi * x)) / np.pi


def predict_antiderivative(model, config, eval_points):
    """用训练好的模型预测测试函数 v(x)=sin(pi*x) 的反导数。"""

    x = np.linspace(0, 1, num=int(config["num_prediction_points"]))
    v_eval = np.sin(np.pi * eval_points.ravel())

    if config["mode"] == "aligned":
        # Cartesian product 模式：branch 输入一个函数，trunk 输入全部预测点。
        branch_input = v_eval[None, :]
    else:
        # unaligned 模式：每个预测点都需要配一份相同的 branch 输入函数。
        branch_input = np.tile(v_eval, (x.size, 1))

    u_pred = np.ravel(model.predict((branch_input, x[:, None])))
    u_true = analytical_solution(x)
    return x, v_eval, u_pred, u_true


def save_prediction_figure(x, eval_points, v_eval, u_pred, u_true, output_dir, mode):
    """保存输入函数、预测解和误差曲线。"""

    l2_error = dde.metrics.l2_relative_error(u_true, u_pred)
    abs_error = np.abs(u_pred - u_true)

    fig = plt.figure(figsize=(7, 9))

    plt.subplot(3, 1, 1)
    plt.title(f"Antiderivative PI-DeepONet ({mode}), L2 relative error={l2_error:.3e}")
    plt.ylabel("v(x)")
    plt.plot(eval_points.ravel(), v_eval, "b--", label="input v(x)")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.25)

    plt.subplot(3, 1, 2)
    plt.ylabel("u(x)")
    plt.plot(x, u_true, "k-", label="exact")
    plt.plot(x, u_pred, "r--", label="predict")
    plt.legend(loc="best", fontsize=8)
    plt.grid(True, alpha=0.25)

    plt.subplot(3, 1, 3)
    plt.ylabel("absolute error")
    plt.semilogy(x, np.maximum(abs_error, 1e-16), "m-")
    plt.xlabel("x")
    plt.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    prediction_path = output_dir / "prediction.png"
    fig.savefig(prediction_path, dpi=150)
    plt.close(fig)

    logger.info("L2 相对误差: %.6e", l2_error)
    logger.info("预测对比图已保存到: %s", prediction_path)


def train_and_save(config):
    """构建数据与网络，训练模型，并保存所有本地结果。"""

    mode = config["mode"]
    output_dir = OUTPUT_ROOT / f"antiderivative_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("DeepXDE 版本: %s", dde.__version__)
    logger.info("DeepXDE 后端: %s", dde.backend.backend_name)
    logger.info("运行模式: %s", mode)
    logger.info("输出目录: %s", output_dir)
    logger.info(
        "训练参数: num_domain=%d, num_boundary=%d, num_eval_points=%d, "
        "num_function=%d, iterations=%d, learning_rate=%.1e, hidden_layers=%s",
        config["num_domain"],
        config["num_boundary"],
        config["num_eval_points"],
        config["num_function"],
        config["iterations"],
        config["learning_rate"],
        config["hidden_layers"],
    )

    _, data_pde = build_pde(config)
    operator_data, eval_points = build_operator_data(config, data_pde)
    net = build_network(config)

    model = dde.Model(operator_data, net)
    model.compile("adam", lr=float(config["learning_rate"]))

    logger.info("开始训练模型...")
    loss_history, train_state = model.train(
        iterations=int(config["iterations"]),
        display_every=int(config["display_every"]),
    )
    logger.info("训练完成，最佳 step: %s", train_state.best_step)

    model_prefix = output_dir / f"antiderivative_{mode}_pideeponet"
    model_path = model.save(str(model_prefix), protocol="backend", verbose=1)
    logger.info("PyTorch 模型已保存到: %s", model_path)

    save_convergence_history(loss_history, output_dir)
    x, v_eval, u_pred, u_true = predict_antiderivative(model, config, eval_points)
    save_prediction_figure(x, eval_points, v_eval, u_pred, u_true, output_dir, mode)

    return {
        "model_path": Path(model_path),
        "output_dir": output_dir,
        "best_step": train_state.best_step,
    }


def main(overrides=None):
    """脚本入口；测试时可传入 overrides 临时缩小训练规模。"""

    config = make_config(overrides)
    return train_and_save(config)


if __name__ == "__main__":
    main()
