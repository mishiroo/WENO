# 分析计算结果的收敛阶并绘图
import numpy as np

from matplotlib import pyplot as plt


def AccuracyAnalysis(N1: int, N2: int, Error1: int, Error2: int) -> float:
    """给定剖分结果，计算收敛阶

    Args:
        N1 (int): 第一次剖分的网格数
        N2 (int): 第二次剖分的网格数
        Error1 (int): 第一次剖分的误差
        Error2 (int): 第二次剖分的误差

    Returns:
        float: 收敛阶
    """
    return np.log(Error1 / Error2) / np.log(N2 / N1)

# test
print(AccuracyAnalysis(101, 201, 4, 1))