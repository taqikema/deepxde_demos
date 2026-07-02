"""DeepXDE 官方 PI-DeepONet 示例：一维 Poisson 方程。

代码来源：
https://deepxde.readthedocs.io/en/latest/demos/operator/poisson.1d.pideeponet.html

本地运行做了两点小调整：
1. 在导入 deepxde 前指定 PyTorch 后端，避免默认寻找未安装的 TensorFlow。
2. 使用 matplotlib 的 Agg 后端并保存图片，避免无图形界面环境中 plt.show() 阻塞。
"""

from pathlib import Path
import logging
import os

# DeepXDE 必须在 import 前设置后端；这里使用当前虚拟环境中已安装的 PyTorch CPU 版。
os.environ.setdefault("DDE_BACKEND", "pytorch")

import matplotlib

# 服务器/WSL 环境通常没有桌面显示器，Agg 可以直接把图保存到文件。
matplotlib.use("Agg")

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
RESULT_FIG_PATH = OUTPUT_DIR / "poisson_1d_pideeponet.png"
LOSS_DATA_PATH = OUTPUT_DIR / "loss_history.dat"
LOSS_FIG_PATH = OUTPUT_DIR / "loss_history.png"

# 将影响精度/耗时的参数集中放在这里，方便继续调参。
NUM_DOMAIN = 200
NUM_BOUNDARY = 2
DEGREE = 3
NUM_EVAL_POINTS = 20
NUM_FUNCTIONS = 300
WIDTH = 64
LATENT_DIM = 64
LBFGS_MAXITER = 3000

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Poisson 方程：-u_xx = f。
def equation(x, y, f):
    # 对网络输出 y 关于空间坐标 x 求二阶导，作为 u_xx。
    dy_xx = dde.grad.hessian(y, x)
    return -dy_xx - f


def analytical_solution(features, xs):
    """计算 -u''=f 且 u(0)=u(1)=0 时的解析解。

    PowerSeries 的特征向量满足 f(x)=sum_i a_i x^i。
    对 -u''=f 两次积分并代入零边界条件，可得
    u(x)=sum_i a_i * (x - x^(i+2)) / ((i+1)(i+2))。
    """

    xs = np.ravel(xs)
    powers = np.arange(features.shape[1])
    denominators = (powers + 1) * (powers + 2)
    basis = (xs[None, :] - xs[None, :] ** (powers[:, None] + 2)) / denominators[:, None]
    return features @ basis


def save_convergence_history(loss_history):
    """保存 DeepXDE 返回的训练历史，用于查看收敛过程。"""

    steps = np.asarray(loss_history.steps)
    train_total = np.asarray([np.sum(loss) for loss in loss_history.loss_train])
    test_total = np.asarray([np.sum(loss) for loss in loss_history.loss_test])
    data = np.column_stack([steps, train_total, test_total])
    np.savetxt(
        LOSS_DATA_PATH,
        data,
        header="step train_total_loss test_total_loss",
        comments="",
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(steps, train_total, "o-", label="Train loss")
    ax.semilogy(steps, test_total, "s-", label="Test loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("DeepXDE training convergence")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(LOSS_FIG_PATH, dpi=150)
    plt.close(fig)

    logger.info("收敛数据已保存到: %s", LOSS_DATA_PATH)
    logger.info("收敛曲线已保存到: %s", LOSS_FIG_PATH)


logger.info("DeepXDE 版本: %s", dde.__version__)
logger.info("DeepXDE 后端: %s", dde.backend.backend_name)
logger.info(
    "DeepXDE 已提供训练历史 API: Model.train(...) 返回 LossHistory/TrainState；"
    "本脚本使用该历史保存收敛曲线，因此无需额外启用 TensorBoard。"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logger.info("输出目录: %s", OUTPUT_DIR)
logger.info(
    "训练参数: num_domain=%d, num_boundary=%d, degree=%d, "
    "num_eval_points=%d, num_function=%d, width=%d, latent_dim=%d, lbfgs_maxiter=%d",
    NUM_DOMAIN,
    NUM_BOUNDARY,
    DEGREE,
    NUM_EVAL_POINTS,
    NUM_FUNCTIONS,
    WIDTH,
    LATENT_DIM,
    LBFGS_MAXITER,
)


# 计算区域为一维区间 [0, 1]。
geom = dde.geometry.Interval(0, 1)


# 零 Dirichlet 边界条件：u(0) = u(1) = 0。
def u_boundary(_):
    return 0


def boundary(_, on_boundary):
    return on_boundary


bc = dde.icbc.DirichletBC(geom, u_boundary, boundary)

# 定义 PDE 采样：增加内部点数可以加强物理约束，通常会降低 PDE 残差。
pde = dde.data.PDE(geom, equation, bc, num_domain=NUM_DOMAIN, num_boundary=NUM_BOUNDARY)

# f(x) 的函数空间：三次多项式空间，N=degree+1 表示系数个数。
space = dde.data.PowerSeries(N=DEGREE + 1)

# branch net 看到的是 f 在这些固定点上的取值；更多采样点可更细地描述源项。
evaluation_points = geom.uniform_points(NUM_EVAL_POINTS, boundary=True)

# PDEOperatorCartesianProd 将 PDE 约束和函数空间组合成 PI-DeepONet 的训练数据。
pde_op = dde.data.PDEOperatorCartesianProd(
    pde,
    space,
    evaluation_points,
    num_function=NUM_FUNCTIONS,
)

# DeepONet 结构：
# branch net 输入 f 的采样值；trunk net 输入空间坐标 x；二者输出维度均为 LATENT_DIM。
dim_x = 1
net = dde.nn.DeepONetCartesianProd(
    [NUM_EVAL_POINTS, WIDTH, WIDTH, LATENT_DIM],
    [dim_x, WIDTH, WIDTH, LATENT_DIM],
    activation="tanh",
    kernel_initializer="Glorot normal",
)

# 构建并训练模型。官方示例使用 L-BFGS；这里提高迭代次数以进一步降低误差。
model = dde.Model(pde_op, net)
dde.optimizers.set_LBFGS_options(maxiter=LBFGS_MAXITER)
model.compile("L-BFGS")
logger.info("开始训练模型...")
losshistory, train_state = model.train(display_every=100)
logger.info("训练完成，最佳 step: %s", train_state.best_step)
save_convergence_history(losshistory)

# 随机采样 3 个不同的源项 f(x)，并预测对应的解 u(x)。
n = 3
features = space.random(n)
fx = space.eval_batch(features, evaluation_points)

x = geom.uniform_points(100, boundary=True)
y = model.predict((fx, x))
u_exact = analytical_solution(features, x)

# 点态相对误差在解析解接近 0 时会数值发散，故用极小量保护分母。
relative_error = np.abs(y - u_exact) / np.maximum(np.abs(u_exact), 1e-8)

# 绘制源项 f(x)、预测解/解析解对比，以及预测值的相对误差。
fig = plt.figure(figsize=(7, 10))
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

plt.subplot(3, 1, 1)
plt.title("Poisson equation: Source term f(x) and solution u(x)")
plt.ylabel("f(x)")
z = np.zeros_like(x)
plt.plot(x, z, "k-", alpha=0.1)

for i in range(n):
    color = colors[i % len(colors)]
    plt.plot(evaluation_points, fx[i], "--", color=color, label=f"f {i + 1}")

plt.legend(loc="best", fontsize=8)

plt.subplot(3, 1, 2)
plt.ylabel("u(x)")
plt.plot(x, z, "k-", alpha=0.1)

for i in range(n):
    color = colors[i % len(colors)]
    plt.plot(x, y[i], "-", color=color, label=f"predict {i + 1}")
    # 用散点展示同一源项对应的解析解，便于和预测曲线直接对比。
    plt.scatter(
        x[::4],
        u_exact[i, ::4],
        s=16,
        marker="o",
        facecolors="none",
        edgecolors=color,
        label=f"exact {i + 1}",
    )

plt.legend(loc="best", fontsize=8, ncol=2)

plt.subplot(3, 1, 3)
plt.ylabel("relative error")

for i in range(n):
    color = colors[i % len(colors)]
    plt.semilogy(x, np.maximum(relative_error[i], 1e-12), "-", color=color, label=f"error {i + 1}")

plt.legend(loc="best", fontsize=8)

plt.xlabel("x")
plt.tight_layout()

fig.savefig(RESULT_FIG_PATH, dpi=150)
plt.close(fig)
logger.info("结果图已保存到: %s", RESULT_FIG_PATH)
