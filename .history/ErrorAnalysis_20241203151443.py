# 分析计算结果的收敛阶 绘制计算效果图
import numpy as np
import pandas as pd
from typing import List
from matplotlib import pyplot as plt


def AccuracyOrder(N1: int, N2: int, Error1: int, Error2: int) -> float:
    """给定两组剖分和计算结果，计算收敛阶

    Args:
        N1 (int): 第一次剖分的网格数
        N2 (int): 第二次剖分的网格数
        Error1 (int): 第一次剖分的误差
        Error2 (int): 第二次剖分的误差

    Returns:
        float: 收敛阶
    """
    return np.log(Error1 / Error2) / np.log(N2 / N1)


def AccuracyOrders(NList: List[int], ErrorList: List[float]) -> np.ndarray:
    """对两组及以上剖分结果进行收敛阶分析 返回长度为len(NList)第一个元素为None的收敛阶数组

    Args:
        NList (List[int]): 网格剖分参数列表
        ErrorList (List[float]): 误差结果列表

    Returns:
        np.ndarray: 收敛阶列表
    """
    assert len(NList) == len(
        ErrorList
    ), f"length of NList and ErrorList must be equal, but {len(NList)} and {len(ErrorList)} are given"
    orderList = np.zeros(len(NList))
    orderList[0] = None
    for i in range(1, len(NList)):
        orderList[i] = AccuracyOrder(
            NList[i], NList[i - 1], ErrorList[i], ErrorList[i - 1]
        )
    return orderList


def SaveError(
    fn: str, NList: List[int], L2ErrorList: List[float], LinfErrorList: List[float]
) -> None:
    """保存误差结果

    Args:
        fn (str): 保存文件名
        NList (List[int]): 网格剖分参数列表
        ErrorList (List[float]): 误差结果列表
    """
    ac_orders_l2 = AccuracyOrders(NList, L2ErrorList)
    ac_orders_linf = AccuracyOrders(NList, LinfErrorList)
    # 保存计算结果
    data = {
        "N": NList,
        "L2Error": L2ErrorList,
        "L2Order": ac_orders_l2,
        "LInfError": LinfErrorList,
        "LInfOrder": ac_orders_linf,
    }
    df = pd.DataFrame(data)
    df.to_excel(fn, index=False)


def L2Error(analytical: List[float], numerical: List[float]) -> float:
    """计算L2误差

    Args:
        analytical (List[float]): 解析解
        numerical (List[float]): 数值解

    Returns:
        float: L2误差
    """
    return np.sqrt(np.sum([(a - n) ** 2 for a, n in zip(analytical, numerical)]))


def LinfError(analytical: List[float], numerical: List[float]) -> float:
    """计算Linf误差

    Args:
        analytical (List[float]): 解析解
        numerical (List[float]): 数值解

    Returns:
        float: Linf误差
    """
    return max([abs(a - n) for a, n in zip(analytical, numerical)])


def Draw(
    mesh: np.ndarray,
    theo_values: np.ndarray,
    weno_values: np.ndarray,
    show=False,
    fn: str = None,
):
    """只绘制剖分为N的精确值与数值解对比图

    Args:
        mesh (np.ndarray): 剖分网格
        theo_values (np.ndarray): 理论值
        weno_values (np.ndarray): WENO求解结果
        show(Bool, optional): 是否显示图形. Defaults to False.
        fn (str, optional): 保存文件名. Defaults to None.
    """
    N = mesh.size - 1
    # 创建图形
    plt.figure(figsize=(8, 6))

    # 绘制理论值
    plt.plot(
        mesh,
        theo_values,
        label="Exact Solution",
        color="blue",
        linestyle="-",
        linewidth=0.5,
    )
    # 绘制WENO数值解
    plt.plot(
        mesh,
        weno_values,
        label="WENO Solution",
        color="green",
        linestyle="--",
        linewidth=0.5,
    )
    # 设置标题和标签

    plt.title(f"WENO Solution at N={N}", fontsize=16)
    plt.xlabel("x", fontsize=14)
    plt.ylabel("u(x)", fontsize=14)

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True)
    if fn is not None:
        plt.savefig(f"figures/{fn}.png")
    else:
        plt.savefig(f"figures/WENO5-N={N}.png")
    # 展示图形
    if show:
        plt.show()
