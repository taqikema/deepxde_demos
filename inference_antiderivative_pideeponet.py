"""PI-DeepONet 推理脚本：加载已训练模型，用新的输入函数进行预测

使用方式：
    conda run -n pinn_env python deeponet/inference_pideeponet.py

前置条件：需要先运行 antiderivative_aligned_pideeponet.py 完成训练，
         模型断点保存在 deeponet/model/ 目录下。
"""

import os
import deepxde as dde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# ============================================================
# 模型文件路径（修改此处以加载不同的训练断点）
# ============================================================
MODEL_PATH = "outputs/antiderivative_aligned/antiderivative_aligned_pideeponet-40000.pt"

# ============================================================
# 1. 重建与训练时完全一致的计算图（必须与训练脚本保持一致）
# ============================================================
transpose = lambda x, y: torch.transpose(x, y[0], y[1])

geom = dde.geometry.TimeDomain(0, 1)

def pde(x, u, v):
    return dde.grad.jacobian(u, x) - v

ic = dde.icbc.IC(geom, lambda _: 0, lambda _, on_initial: on_initial)
pde_data = dde.data.PDE(geom, pde, ic, num_domain=20, num_boundary=2, num_test=40)

func_space = dde.data.GRF(length_scale=0.2)

eval_pts = np.linspace(0, 1, num=50)[:, None]
data = dde.data.PDEOperatorCartesianProd(
    pde_data, func_space, eval_pts, 1000, num_test=100, batch_size=100
)

net = dde.nn.DeepONetCartesianProd(
    [50, 128, 128, 128],
    [1, 128, 128, 128],
    "tanh",
    "Glorot normal",
)

def zero_ic(inputs, outputs):
    return outputs * transpose(inputs[1], [1, 0])

net.apply_output_transform(zero_ic)

model = dde.Model(data, net)
model.compile("adam", lr=0.0005)

# ============================================================
# 2. 加载已训练的模型权重
# ============================================================
print(f"加载模型: {MODEL_PATH}")
model.restore(MODEL_PATH, verbose=1)
print("模型加载完成！\n")

# ============================================================
# 3. 定义测试函数集合并批量推理
# ============================================================
x = np.linspace(0, 1, num=50)

test_functions = {
    # v(x) = sin(πx)，解析解：u(x) = 1/π - cos(πx)/π
    "sin(pi*x)": {
        "v": np.sin(np.pi * eval_pts).T,
        "u_true": 1 / np.pi - np.cos(np.pi * x) / np.pi,
    },
    # v(x) = x，解析解：u(x) = x²/2
    "x": {
        "v": eval_pts.T,
        "u_true": x**2 / 2,
    },
    # v(x) = x²，解析解：u(x) = x³/3
    "x^2": {
        "v": (eval_pts**2).T,
        "u_true": x**3 / 3,
    },
    # v(x) = cos(πx)，解析解：u(x) = sin(πx)/π
    "cos(pi*x)": {
        "v": np.cos(np.pi * eval_pts).T,
        "u_true": np.sin(np.pi * x) / np.pi,
    },
    # v(x) = exp(-x)，解析解：u(x) = 1 - exp(-x)
    "exp(-x)": {
        "v": np.exp(-eval_pts).T,
        "u_true": 1 - np.exp(-x),
    },
}

# --- 批量推理并绘图 ---
n_funcs = len(test_functions)
cols = min(3, n_funcs)
rows = (n_funcs + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = np.atleast_1d(axes).flatten()

results = {}
for ax, (name, func) in zip(axes, test_functions.items()):
    v_input = func["v"]
    u_true = func["u_true"]

    # 模型预测
    u_pred = np.ravel(model.predict((v_input, x[:, None])))

    # 计算误差
    l2_err = dde.metrics.l2_relative_error(u_true, u_pred)
    results[name] = l2_err

    # 绘图
    ax.plot(x, u_true, "k", label="True")
    ax.plot(x, u_pred, "r--", label="Predicted")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title(f"v(x) = {name}\nL2 rel. err = {l2_err:.4e}")
    ax.legend()

# 隐藏多余的子图
for ax in axes[n_funcs:]:
    ax.set_visible(False)

plt.tight_layout()
os.makedirs("deeponet", exist_ok=True)
plt.savefig("deeponet/inference_results.png", dpi=150, bbox_inches="tight")
print("推理结果图已保存至 deeponet/inference_results.png\n")

# --- 控制台输出所有误差 ---
print("=" * 55)
print(f"{'测试函数':<20s} {'L2 相对误差':>15s}")
print("=" * 55)
for name, err in results.items():
    print(f"v(x) = {name:<15s} {err:>15.6e}")
print("=" * 55)

# 平均误差
avg_err = np.mean(list(results.values()))
print(f"{'平均误差':<20s} {avg_err:>15.6e}")
