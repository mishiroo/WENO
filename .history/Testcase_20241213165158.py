# 构建测试用例
import WENO
from typing import Callable
import numpy as np


def shockwave_fun(x: np.ndarray) -> np.ndarray:
    """间断函数

    Args:
        x (np.ndarray): 传入网格点

    Returns:
        np.ndarray: 对应函数值
    """
    return np.where(x <= 0.5, 1, 0)


class TestCase:
    def __init__(self, mesh: WENO.Mesh):
        self.mesh = mesh  # 网格类 不是具体的网格
        self.N = self.mesh.N  # 单元格数
        ...

    def getTheoValues(self):
        # 获取网格点上的理论值
        ...

    def getAverageValues(self, func_name: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """获取每个单元格的的平均值

        Args:
            func_name (Callable[[np.ndarray], np.ndarray]): _description_

        Returns:
            np.ndarray: _description_
        """
        # 获取每个单元格的的平均值
        ...
