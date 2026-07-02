"""DeepXDE 官方 PI-DeepONet 示例：二维平流方程算子。

代码来源：
https://raw.githubusercontent.com/lululxvi/deepxde/master/examples/operator/advection_aligned_pideeponet_2d.py
https://raw.githubusercontent.com/lululxvi/deepxde/master/examples/operator/advection_unaligned_pideeponet_2d.py

与一维版本的关键区别：
1. 将时间 t 视为第二个空间坐标，使用 dde.geometry.Rectangle([0,0],[1,1]) 作为计算域。
2. 初值条件通过 DirichletBC on x[1]==0 实现（代替 TimePDE + IC）。
3. 使用 dde.data.PDE 代替 dde.data.TimePDE。

本地版本做了以下整理：
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

# ============================================================================
# 控制开关
# ============================================================================
TRAIN_MODE = True                      # True: 训练模式；False: 载入已有模型做预测
MODEL_PATH = "outputs/advection_2d_aligned/advection_2d_aligned_pideeponet-30000.pt"


# 将影响训练和数据格式的参数集中放在这里，方便继续调参。
CONFIG = {
    # "aligned" 使用 Cartesian product 数据；"unaligned" 使用逐点配对数据。
    "mode": "aligned",
    "num_domain": 200,
    "num_boundary": 200,
    "kernel": "ExpSineSquared",
    "length_scale": 1.0,
    "num_eval_points": 50,
    "num_function": 1000,
    "num_test_aligned": 100,
    "num_test_unaligned": 1000,
    "batch_size": 32,
    "hidden_layers": [128, 128, 128],
    "activation": "tanh",
    "kernel_initializer": "Glorot normal",
    "learning_rate": 5e-4,
    "iterations": 30000,
    "display_every": 1000,
    "num_prediction_points": 100,
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
        "num_eval_points",
        "num_function",
        "iterations",
        "num_prediction_points",
    ]
    for key in positive_int_keys:
        if int(config[key]) <= 0:
            raise ValueError(f'CONFIG["{key}"] 必须是正整数。')


# ============================================================================
# 1. PDE 定义
# ============================================================================


def advection_pde(x, y, _v):
    """二维平流方程 u_t + u_x = 0。

    此处将时间 t 视为第二个空间坐标 x[1]。
    dy_x = ∂u/∂x (j=0 为空间方向)
    dy_t = ∂u/∂t (j=1 为时间方向)
    """

    dy_x = dde.grad.jacobian(y, x, j=0)
    dy_t = dde.grad.jacobian(y, x, j=1)
    return dy_t + dy_x


def func_ic(_x, v):
    """初值条件：u(x, 0) = v(x)。

    注意：与一维版本不同，这里使用 DirichletBC 在 x[1]==0 边界上施加初值，
    因为时间 t 已作为第二个空间坐标处理。
    """

    return v


def boundary_on_initial(x, on_boundary):
    """判断点是否位于初值边界（t = x[1] = 0）。"""

    return on_boundary and np.isclose(x[1], 0)


# ============================================================================
# 2. 构建计算域、PDE 和算子数据
# ============================================================================


def build_pde(config):
    """构造二维矩形区域 [0,1]×[0,1] 和初值约束。"""

    # 矩形区域 [0,1]×[0,1]：x 轴为空间，y 轴为时间
    geom = dde.geometry.Rectangle([0, 0], [1, 1])

    # 初值条件施加在 y=0 (即 t=0) 边界上
    ic = dde.icbc.DirichletBC(geom, func_ic, boundary_on_initial)

    data_pde = dde.data.PDE(
        geom,
        advection_pde,
        ic,
        num_domain=int(config["num_domain"]),
        num_boundary=int(config["num_boundary"]),
    )
    return data_pde


def build_operator_data(config, data_pde):
    """根据 aligned / unaligned 模式构造算子学习数据。"""

    eval_points = np.linspace(0, 1, num=int(config["num_eval_points"]))[:, None]

    # ExpSineSquared 周期核适合生成周期输入函数 v(x)
    func_space = dde.data.GRF(
        kernel=config["kernel"],
        length_scale=float(config["length_scale"]),
    )

    if config["mode"] == "aligned":
        # function_variables=[0] 表示输入函数 v 只依赖空间变量 x（不依赖时间）
        data = dde.data.PDEOperatorCartesianProd(
            data_pde,
            func_space,
            eval_points,
            int(config["num_function"]),
            function_variables=[0],
            num_test=int(config["num_test_aligned"]),
            batch_size=int(config["batch_size"]),
        )
    else:
        data = dde.data.PDEOperator(
            data_pde,
            func_space,
            eval_points,
            int(config["num_function"]),
            function_variables=[0],
            num_test=int(config["num_test_unaligned"]),
        )

    return data, eval_points


# ============================================================================
# 3. 网络构建
# ============================================================================


def periodic_features(x):
    """把空间坐标展开成周期特征，使网络更容易表达周期解。

    trunk 输入为 (x, t) 二维，输出为 5 维特征：
    [cos(2πx), sin(2πx), cos(4πx), sin(4πx), t]
    """

    x_space, t = x[:, :1], x[:, 1:]
    angle = x_space * 2 * np.pi
    return torch.cat(
        [
            torch.cos(angle),
            torch.sin(angle),
            torch.cos(2 * angle),
            torch.sin(2 * angle),
            t,
        ],
        dim=1,
    )


def build_network(config):
    """根据数据格式构造匹配的 DeepONet 网络。"""

    branch_layers = [int(config["num_eval_points"]), *config["hidden_layers"]]

    # periodic_features 会把原始 trunk 输入 (x, t) 映射成 5 维特征
    trunk_layers = [5, *config["hidden_layers"]]

    if config["mode"] == "aligned":
        net = dde.nn.DeepONetCartesianProd(
            branch_layers,
            trunk_layers,
            config["activation"],
            config["kernel_initializer"],
        )
    else:
        net = dde.nn.DeepONet(
            branch_layers,
            trunk_layers,
            config["activation"],
            config["kernel_initializer"],
        )

    net.apply_feature_transform(periodic_features)
    return net


# ============================================================================
# 4. 损失曲线保存
# ============================================================================


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
    ax.set_title("Advection PI-DeepONet convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_fig_path, dpi=150)
    plt.close(fig)

    logger.info("收敛数据已保存到: %s", loss_data_path)
    logger.info("收敛曲线已保存到: %s", loss_fig_path)


# ============================================================================
# 5. 预测与可视化
# ============================================================================


def analytical_solution(x, t):
    """初值 v(x)=sin(2*pi*x) 时，平流方程解析解 u(x,t)=sin(2*pi*(x-t))。"""

    return np.sin(2 * np.pi * (x - t[:, None]))


def predict_advection(model, config, eval_points):
    """用训练好的模型预测周期初值 v(x)=sin(2*pi*x) 对应的时空解。

    二维版本中，trunk 输入为 (x, t) 的网格点对。
    """

    n = int(config["num_prediction_points"])
    x = np.linspace(0, 1, num=n)
    t = np.linspace(0, 1, num=n)
    x_mesh, t_mesh = np.meshgrid(x, t)
    trunk_input = np.vstack((np.ravel(x_mesh), np.ravel(t_mesh))).T
    branch_values = np.sin(2 * np.pi * eval_points.ravel())

    if config["mode"] == "aligned":
        # aligned 模式：branch 输入 shape 为 (1, num_eval_points)
        branch_input = branch_values[None, :]
    else:
        # unaligned 模式：branch 输入需复制到与 trunk 相同数量的行
        branch_input = np.tile(branch_values, (trunk_input.shape[0], 1))

    u_pred = model.predict((branch_input, trunk_input)).reshape((n, n))
    u_true = analytical_solution(x, t)
    return x, t, branch_values, u_pred, u_true


def save_prediction_figure(
    x, t, eval_points, branch_values, u_pred, u_true, output_dir, mode
):
    """保存初值函数、真实解、预测解和绝对误差图。"""

    l2_error = dde.metrics.l2_relative_error(u_true, u_pred)
    abs_error = np.abs(u_pred - u_true)
    extent = [x.min(), x.max(), t.min(), t.max()]

    fig = plt.figure(figsize=(9, 8))

    # 图 1：初值函数 v(x)
    ax0 = plt.subplot(2, 2, 1)
    ax0.set_title("Initial function v(x)")
    ax0.plot(eval_points.ravel(), branch_values, "b--")
    ax0.set_xlabel("x")
    ax0.set_ylabel("v(x)")
    ax0.grid(True, alpha=0.25)

    # 图 2：真实解 u(x, t)
    ax1 = plt.subplot(2, 2, 2)
    im1 = ax1.imshow(u_true, extent=extent, origin="lower", aspect="auto")
    ax1.set_title("Exact u(x,t)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    fig.colorbar(im1, ax=ax1, shrink=0.85)

    # 图 3：预测解 u(x, t)
    ax2 = plt.subplot(2, 2, 3)
    im2 = ax2.imshow(u_pred, extent=extent, origin="lower", aspect="auto")
    ax2.set_title(f"Predicted u(x,t), {mode}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    fig.colorbar(im2, ax=ax2, shrink=0.85)

    # 图 4：逐点绝对误差 |u_pred - u_true|
    ax3 = plt.subplot(2, 2, 4)
    im3 = ax3.imshow(abs_error, extent=extent, origin="lower", aspect="auto")
    ax3.set_title(f"Absolute error, L2={l2_error:.3e}")
    ax3.set_xlabel("x")
    ax3.set_ylabel("t")
    fig.colorbar(im3, ax=ax3, shrink=0.85)

    fig.tight_layout()
    prediction_path = output_dir / "prediction.png"
    fig.savefig(prediction_path, dpi=150)
    plt.close(fig)

    logger.info("L2 相对误差: %.6e", l2_error)
    logger.info("预测对比图已保存到: %s", prediction_path)


# ============================================================================
# 6. 主流程：训练/载入 → 预测 → 可视化
# ============================================================================


def run_model(config):
    """构建数据与网络，根据 TRAIN_MODE 决定训练或载入模型，然后做预测和可视化。"""

    mode = config["mode"]
    output_dir = OUTPUT_ROOT / f"advection_2d_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("DeepXDE 版本: %s", dde.__version__)
    logger.info("DeepXDE 后端: %s", dde.backend.backend_name)
    logger.info("运行模式: %s (2D)", mode)
    logger.info("输出目录: %s", output_dir)
    logger.info("TRAIN_MODE: %s", TRAIN_MODE)

    data_pde = build_pde(config)
    operator_data, eval_points = build_operator_data(config, data_pde)
    net = build_network(config)

    model = dde.Model(operator_data, net)
    model.compile("adam", lr=float(config["learning_rate"]))

    if TRAIN_MODE:
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
        logger.info("开始训练模型...")
        loss_history, train_state = model.train(
            iterations=int(config["iterations"]),
            display_every=int(config["display_every"]),
        )
        logger.info("训练完成，最佳 step: %s", train_state.best_step)

        model_prefix = output_dir / f"advection_2d_{mode}_pideeponet"
        model.save(str(model_prefix), protocol="backend", verbose=1)

        save_convergence_history(loss_history, output_dir)

        best_step = train_state.best_step
    else:
        logger.info("载入已有模型: %s", MODEL_PATH)
        model.restore(MODEL_PATH)
        best_step = None

    x, t, branch_values, u_pred, u_true = predict_advection(
        model, config, eval_points
    )
    save_prediction_figure(
        x, t, eval_points, branch_values, u_pred, u_true, output_dir, mode
    )

    return {
        "output_dir": output_dir,
        "best_step": best_step,
    }


def main(overrides=None):
    """脚本入口；测试时可传入 overrides 临时缩小训练规模。"""

    config = make_config(overrides)
    return run_model(config)


if __name__ == "__main__":
    main()
