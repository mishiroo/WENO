from typing import List
from matplotlib import pyplot as plt
import numpy as np
# 完成WENO计算结果的绘图
def Draw(mesh: np.ndarray, theo_values: np.ndarray, weno_values: np.ndarray):
    """只绘制剖分为N的精确值与数值解对比图

    Args:
        mesh (np.ndarray): 剖分网格
        theo_values (np.ndarray): 理论值
        weno_values (np.ndarray): WENO求解结果
    """
    N = mesh.size - 1 
    # 创建图形
    plt.figure(figsize=(8, 6))

    # 绘制理论值
    plt.plot(
        mesh, theo_values, label="Exact Solution", color="b", linestyle="-", linewidth=2
    )

    # 绘制WENO数值解
    plt.plot(
        mesh, weno_values, label="WENO Solution", color="r", linestyle="o", linewidth=2
    )

    # 设置标题和标签
    plt.title(f"WENO Solution at N={N}", fontsize=16)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("u(x)", fontsize=14)

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True)
    plt.savefig(f"figures/WENO-N={}.png")
    # 展示图形
    plt.show()

def DrawAll(NList: List[int], theo_values: np.ndarray, weno_values: np.ndarray):
    plt.figure(figsize=(10, 6))
    for i in range(len(NList)):
        plt.plot(theo_values[i], label=f'N={NList[i]}')
        plt.plot(weno_values[i], label=f'N={NList[i]}-WENO')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('WENO Reconstructed Solution')
    plt.grid(True)
