# 分析计算结果的收敛阶
import numpy as np
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
    """对两组及以上剖分结果进行收敛阶分析 返回长度为len(NList)-1的收敛阶数组

    Args:
        NList (List[int]): 网格剖分参数列表
        ErrorList (List[float]): 误差结果列表

    Returns:
        np.ndarray: 收敛阶列表
    """
    assert len(NList) == len(
        ErrorList
    ), f"length of NList and ErrorList must be equal, but {len(NList)} and {len(ErrorList)} are given"
    orderList = np.zeros(len(NList) - 1)
    for i in range(len(NList) - 1):
        orderList[i] = AccuracyOrder(NList[i], NList[i + 1], ErrorList[i], ErrorList[i + 1])
    return orderList

