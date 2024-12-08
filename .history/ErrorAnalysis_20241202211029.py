# 分析计算结果的收敛阶
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
        orderList[i] = AccuracyOrder(NList[i], NList[i - 1], ErrorList[i], ErrorList[i - 1])
    return orderList

def SaveError(fn: str, NList: List[int], ErrorList: List[float]) -> None:
    """保存误差结果

    Args:
        fn (str): 保存文件名
        NList (List[int]): 网格剖分参数列表
        ErrorList (List[float]): 误差结果列表
    """
    # 保存计算结果
    data = {
        "N": NList,
        "L2Error": L2ErrorList,
        "L2Order": ac_orders_l2,
        "LInfError": LinfErrorList,
        "LInfOrder": ac_orders_linf,
    }
    df = pd.DataFrame(data)
    df.to_excel("output.xlsx", index=False)
