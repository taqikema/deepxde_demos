"""DeepXDE 官方 PI-DeepONet 示例：扩散-反应方程算子。

代码来源：
https://raw.githubusercontent.com/lululxvi/deepxde/master/examples/operator/diff_rec_aligned_pideeponet.py
https://raw.githubusercontent.com/lululxvi/deepxde/master/examples/operator/diff_rec_aligned_zcs_pideeponet.py

本地版本做了四点整理：
1. 在同一份脚本中通过 CONFIG["use_zcs"] 切换普通自动微分 / ZCS 自动微分。
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


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "outputs"

# 将影响训练、数据和微分方式的参数集中放在这里，方便继续调参。
CONFIG = {
    # False 使用 dde.grad；True 使用 dde.zcs.LazyGrad。
    "use_zcs": True,
    "diffusion": 0.01,
    "reaction": 0.01,
    "num_domain": 200,
    "num_boundary": 40,
    "num_initial": 20,
    "num_test_pde": 500,
    "length_scale": 0.2,
    "num_eval_points": 50,
    "num_function": 1000,
    "num_test": 100,
    "batch_size": 50,
    "hidden_layers": [128, 128, 128],
    "activation": "tanh",
    "kernel_initializer": "Glorot normal",
    "learning_rate": 5e-4,
    "iterations": 20000,
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
    """检查配置中的关键参数，避免 DeepXDE 在更深处报晦涩错误。"""

    positive_int_keys = [
        "num_domain",
        "num_boundary",
        "num_initial",
        "num_test_pde",
        "num_eval_points",
        "num_function",
        "num_test",
        "batch_size",
        "iterations",
        "num_prediction_points",
    ]
    for key in positive_int_keys:
        if int(config[key]) <= 0:
            raise ValueError(f'CONFIG["{key}"] 必须是正整数。')

    if float(config["diffusion"]) <= 0:
        raise ValueError('CONFIG["diffusion"] 必须为正数。')


def solve_adr(xmin, xmax, tmin, tmax, k, v, g, dg, f, u0, nx, nt):
    """求解一维 ADR 方程，用于生成预测对比图的参考解。

    该函数内联自 DeepXDE examples/operator/ADR_solver.py。
    """

    x = np.linspace(xmin, xmax, nx)
    t = np.linspace(tmin, tmax, nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h**2

    d1 = np.eye(nx, k=1) - np.eye(nx, k=-1)
    d2 = -2 * np.eye(nx) + np.eye(nx, k=-1) + np.eye(nx, k=1)
    d3 = np.eye(nx - 2)
    k = k(x)
    m = -np.diag(d1 @ k) @ d1 - 4 * np.diag(k) @ d2
    m_bond = 8 * h2 / dt * d3 + m[1:-1, 1:-1]
    v = v(x)
    v_bond = 2 * h * np.diag(v[1:-1]) @ d1[1:-1, 1:-1] + 2 * h * np.diag(v[2:] - v[: nx - 2])
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * d3 - m[1:-1, 1:-1] - v_bond
    f = f(x[:, None], t)

    u = np.zeros((nx, nt))
    u[:, 0] = u0(x)
    for i in range(nt - 1):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        a = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1, i] + 0.5 * f[1:-1, i + 1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u[1:-1, i + 1] = np.linalg.solve(a, b1 + b2)
    return x, t, u


def make_pde_function(config):
    """根据配置创建 PDE 函数，普通自动微分和 ZCS 自动微分只差导数计算方式。"""

    diffusion = float(config["diffusion"])
    reaction = float(config["reaction"])

    if config["use_zcs"]:

        def pde_fn(x, y, source):
            grad_y = dde.zcs.LazyGrad(x, y)
            dy_t = grad_y.compute((0, 1))
            dy_xx = grad_y.compute((2, 0))
            return dy_t - diffusion * dy_xx + reaction * y**2 - source

    else:

        def pde_fn(x, y, source):
            dy_t = dde.grad.jacobian(y, x, j=1)
            dy_xx = dde.grad.hessian(y, x, j=0)
            return dy_t - diffusion * dy_xx + reaction * y**2 - source

    return pde_fn


def build_pde(config):
    """构造空间-时间区域、零边界条件和零初值条件。"""

    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    bc = dde.icbc.DirichletBC(geomtime, lambda _: 0, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, lambda _: 0, lambda _, on_initial: on_initial)
    pde_fn = make_pde_function(config)

    data_pde = dde.data.TimePDE(
        geomtime,
        pde_fn,
        [bc, ic],
        num_domain=int(config["num_domain"]),
        num_boundary=int(config["num_boundary"]),
        num_initial=int(config["num_initial"]),
        num_test=int(config["num_test_pde"]),
    )
    return geomtime, data_pde


def build_operator_data(config, data_pde):
    """构造 Cartesian product 格式的 PI-DeepONet 训练数据。"""

    eval_points = np.linspace(0, 1, num=int(config["num_eval_points"]))[:, None]
    func_space = dde.data.GRF(length_scale=float(config["length_scale"]))
    operator_cls = dde.zcs.PDEOperatorCartesianProd if config["use_zcs"] else dde.data.PDEOperatorCartesianProd
    data = operator_cls(
        data_pde,
        func_space,
        eval_points,
        int(config["num_function"]),
        function_variables=[0],
        num_test=int(config["num_test"]),
        batch_size=int(config["batch_size"]),
    )
    return data, func_space, eval_points


def build_network(config):
    """构造 DeepONetCartesianProd 网络。"""

    branch_layers = [int(config["num_eval_points"]), *config["hidden_layers"]]
    trunk_layers = [2, *config["hidden_layers"]]
    return dde.nn.DeepONetCartesianProd(
        branch_layers,
        trunk_layers,
        config["activation"],
        config["kernel_initializer"],
    )


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
    ax.set_title("Diffusion-reaction PI-DeepONet convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_fig_path, dpi=150)
    plt.close(fig)

    logger.info("收敛数据已保存到: %s", loss_data_path)
    logger.info("收敛曲线已保存到: %s", loss_fig_path)


def predict_diff_rec(model, config, func_space, eval_points):
    """随机采样一个源项，并预测对应的扩散-反应方程解。"""

    n = int(config["num_prediction_points"])
    func_features = func_space.random(1)
    xs = np.linspace(0, 1, num=n)[:, None]
    source_on_grid = func_space.eval_batch(func_features, xs)[0]

    diffusion = float(config["diffusion"])
    reaction = float(config["reaction"])
    x, t, u_true = solve_adr(
        0,
        1,
        0,
        1,
        lambda x_: diffusion * np.ones_like(x_),
        lambda x_: np.zeros_like(x_),
        lambda u: reaction * u**2,
        lambda u: 2 * reaction * u,
        lambda _x, t_: np.tile(source_on_grid[:, None], (1, len(t_))),
        lambda x_: np.zeros_like(x_),
        n,
        n,
    )
    u_true = u_true.T

    branch_input = func_space.eval_batch(func_features, eval_points)
    x_mesh, t_mesh = np.meshgrid(x, t)
    trunk_input = np.vstack((np.ravel(x_mesh), np.ravel(t_mesh))).T
    u_pred = model.predict((branch_input, trunk_input)).reshape((n, n))
    return x, t, source_on_grid, u_pred, u_true


def save_prediction_figure(x, t, source_on_grid, u_pred, u_true, output_dir, label):
    """保存源项、真实解、预测解和误差图。"""

    l2_error = dde.metrics.l2_relative_error(u_true, u_pred)
    abs_error = np.abs(u_pred - u_true)
    extent = [x.min(), x.max(), t.min(), t.max()]

    fig = plt.figure(figsize=(9, 8))

    ax0 = plt.subplot(2, 2, 1)
    ax0.set_title("Source function f(x)")
    ax0.plot(x, source_on_grid, "b--")
    ax0.set_xlabel("x")
    ax0.set_ylabel("f(x)")
    ax0.grid(True, alpha=0.25)

    ax1 = plt.subplot(2, 2, 2)
    im1 = ax1.imshow(u_true, extent=extent, origin="lower", aspect="auto")
    ax1.set_title("Reference u(x,t)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    fig.colorbar(im1, ax=ax1, shrink=0.85)

    ax2 = plt.subplot(2, 2, 3)
    im2 = ax2.imshow(u_pred, extent=extent, origin="lower", aspect="auto")
    ax2.set_title(f"Predicted u(x,t), {label}")
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    fig.colorbar(im2, ax=ax2, shrink=0.85)

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


def train_and_save(config):
    """构建数据与网络，训练模型，并保存所有本地结果。"""

    label = "zcs" if config["use_zcs"] else "standard"
    output_dir = OUTPUT_ROOT / ("diff_rec_aligned_zcs" if config["use_zcs"] else "diff_rec_aligned")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("DeepXDE 版本: %s", dde.__version__)
    logger.info("DeepXDE 后端: %s", dde.backend.backend_name)
    logger.info("使用 ZCS: %s", config["use_zcs"])
    logger.info("输出目录: %s", output_dir)
    logger.info(
        "训练参数: num_domain=%d, num_boundary=%d, num_initial=%d, "
        "num_eval_points=%d, num_function=%d, iterations=%d, learning_rate=%.1e",
        config["num_domain"],
        config["num_boundary"],
        config["num_initial"],
        config["num_eval_points"],
        config["num_function"],
        config["iterations"],
        config["learning_rate"],
    )

    _, data_pde = build_pde(config)
    operator_data, func_space, eval_points = build_operator_data(config, data_pde)
    net = build_network(config)

    model_cls = dde.zcs.Model if config["use_zcs"] else dde.Model
    model = model_cls(operator_data, net)
    model.compile("adam", lr=float(config["learning_rate"]))

    logger.info("开始训练模型...")
    loss_history, train_state = model.train(
        iterations=int(config["iterations"]),
        display_every=int(config["display_every"]),
    )
    logger.info("训练完成，最佳 step: %s", train_state.best_step)

    model_prefix = output_dir / f"diff_rec_aligned_{label}_pideeponet"
    model_path = model.save(str(model_prefix), protocol="backend", verbose=1)
    logger.info("PyTorch 模型已保存到: %s", model_path)

    save_convergence_history(loss_history, output_dir)
    x, t, source_on_grid, u_pred, u_true = predict_diff_rec(model, config, func_space, eval_points)
    save_prediction_figure(x, t, source_on_grid, u_pred, u_true, output_dir, label)

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
