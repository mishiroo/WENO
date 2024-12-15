# 构建测试用例
import WENO
import GaussIntegrate
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

# 间断问题高斯积分求不对 这里用解析解计算
def shockwave_fun_avg(x: np.ndarray) -> np.ndarray:


def sin_fun(x: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x)

def exp_fun(x: np.ndarray) -> np.ndarray:
    return np.exp(x)

class TestCase:
    def __init__(self, mesh: WENO.Mesh):
        self.mesh = mesh.mesh  # 具体的网格 长度为N+1
        self.N = mesh.N  # 单元格数
        self._avg_values = np.zeros(self.N)  # 初始化平均值

    def getAverageValues(
        self, func_name: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        # 利用高斯积分求解区域均值
        for i in range(self.N):
            self._avg_values[i] = GaussIntegrate.gauss_integrate(
                func_name, [self.mesh[i], self.mesh[i + 1]]
            ) / (self.mesh[i + 1] - self.mesh[i])
        return self._avg_values
    

    def getTheoValues(
        self, func_name: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """获取每个网格点的理论值

        Args:
            func_name (Callable[[np.ndarray], np.ndarray]): 定义的函数
                需要实现网格点数组到网格点对应函数值数组的映射

        Returns:
            np.ndarray: 理论值
        """
        return func_name(self.mesh)


# 测试用例
# if __name__ == "__main__":
#     mesh = WENO.Mesh(0, 1, 100)
#     test_case = TestCase(mesh)
#     print(test_case.getAverageValues(shockwave_fun))
#     print(test_case.getTheoValues(shockwave_fun))
