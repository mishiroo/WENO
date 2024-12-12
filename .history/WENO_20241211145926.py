import tqdm
import numpy as np
from typing import List
# 1D dimension
class WENO1D:

    def __init__(self, mesh: np.ndarray, avg_values: np.ndarray, order=5) -> None:
        """初始化WENO类

        Args:
            mesh (np.ndarray): 网格剖分[lb,...ub]
            avg_values (np.ndarray): 网格剖分[lb,...ub]
            order (int, optional): WENO算法阶数. Defaults to 5.
        """
        assert np.size(mesh) - 1 == np.size(
            avg_values
        ), "len(mesh) == len(avg_values) + 1"
        self.avg_values = avg_values
        self.mesh = mesh

        self.order = order
        self.lb = mesh[0]
        self.ub = mesh[-1]
        self.N = len(mesh) - 1  # Num of cells
        self.k = int((order + 1) / 2)  # Num of stencils
        # 查表得到模板组合线性权
        if order == 1:
            self.weights_d = [1]
        elif order == 3:
            self.weights_d = [2 / 3, 1 / 3]
        elif order == 5:
            self.weights_d = [1 / 10, 3 / 5, 3 / 10]
            # 查表得到模板组合系数
            self.C = np.array(
                [
                    [1 / 3, -7 / 6, 11 / 6],
                    [-1 / 6, 5 / 6, 1 / 3],
                    [1 / 3, 5 / 6, -1 / 6],
                    [11 / 6, -7 / 6, 1 / 3],
                ]
            )
        else:
            raise ValueError("order must be 1, 2 or 3")  # 其他阶的系数论文没给，先不写

    def StencilReconstruct(self, index: int) -> List[float]:
        """根据模板进行重构 精度为k阶

        Args:
            index (int): 计算节点的位置索引

        Returns:
            List[float]: 不同模板上的重构结果
        """
        stencil_index = index - 1
        stencil_recons = np.zeros(self.k)  # 存储不同模板重构结果的值
        if stencil_index - self.k + 1 >= 0 and stencil_index + self.k - 1 <= self.N - 1:
            for i in range(self.k):
                stencil_recons[i] = np.inner(
                    self.C[i],
                    self.avg_values[
                        stencil_index - self.k + 1 + i : stencil_index + i + 1
                    ],
                )

        elif stencil_index - self.k + 1 < 0:
            for i in range(self.k):
                stencil_recons[i] = np.inner(
                    self.C[-1],
                    self.avg_values[stencil_index+1 : stencil_index + self.k+1],
                )

        elif stencil_index + self.k - 1 > self.N - 1:
            for i in range(self.k):
                stencil_recons[i] = np.inner(
                    self.C[0],
                    self.avg_values[stencil_index - self.k + 1 : stencil_index + 1],
                )
        return stencil_recons


    def SmoothIndicators(self, index: int) -> np.ndarray:
        """计算指示器

        Args:
            index (int): 索引

        Returns:
            np.ndarray: 平滑指示函数
        """
        assert self.k == 2 or self.k == 3, "SmoothIndicators only support k=3 NOW!"
        # 处理非法值
        if index < self.k - 1 or index > self.N - self.k:
            return self.weights_d
        else:
            if self.k == 2:
                beta0 = self.avg_values[index + 1] - self.avg_values[index]
                beta1 = self.avg_values[index] - self.avg_values[index - 1]
                return [beta0, beta1]
            elif self.k == 3:
                beta0 = (
                    13
                    / 12
                    * (
                        self.avg_values[index + 1]
                        - 2 * self.avg_values[index + 1]
                        + self.avg_values[index + 2]
                    )
                    ** 2
                )
                +1 / 4 * (
                    3 * self.avg_values[index]
                    - 4 * self.avg_values[index + 2]
                    + self.avg_values[index + 2]
                ) ** 2
                beta1 = (
                    13
                    / 12
                    * (
                        self.avg_values[index - 1]
                        - 2 * self.avg_values[index]
                        + self.avg_values[index + 1]
                    )
                    ** 2
                )
                +1 / 4 * (self.avg_values[index - 1] - self.avg_values[index + 1]) ** 2
                beta2 = (
                    13
                    / 12
                    * (
                        self.avg_values[index - 2]
                        - 2 * self.avg_values[index - 1]
                        + self.avg_values[index]
                    )
                    ** 2
                )
                +1 / 4 * (
                    3 * self.avg_values[index]
                    - 4 * self.avg_values[index - 1]
                    + self.avg_values[index - 2]
                ) ** 2
            return [beta0, beta1, beta2]

    def FormWeights(self, index: int, eps=1e-6) -> np.ndarray:
        """计算归一化权重

        Args:
            index (int): 索引

        Returns:
            np.ndarray: 权重
        """
        alpha = np.zeros(self.k)
        beta = self.SmoothIndicators(index)
        for i in range(self.k):
            alpha[i] = self.weights_d[i] / (eps + beta[i]) ** 2
        return alpha / np.sum(alpha)

    def WENOReconstruct(self, index: int, eps=1e-6) -> float:
        """WENO加权后的重构 精度为2k-1阶

        Args:
            index (int): 索引
            eps (_type_, optional): eps参数. Defaults to 1e-6.

        Returns:
            float: 插值结果
        """
        weights = self.FormWeights(index, eps=eps)
        return np.inner(self.weights_d, self.StencilReconstruct(index))


class Mesh:
    def __init__(self, lb: float, ub: float, N: int) -> None:
        """初始化网格

        Args:
            lb (float): 左边界
            ub (float): 右边界
            N (int): cells数量
        """
        self.lb = lb
        self.ub = ub
        self.N = N
        self.mesh = np.linspace(lb, ub, N + 1)
        self.avg_values = np.zeros(N)  # Initialize avg_values

    def BindAverageValues(self, avg_values: List[float]) -> None:
        """绑定平均值

        Args:
            avg_values (List[float]): 平均值
        """
        assert len(avg_values) == self.N, "avg_values length must be N"
        self.avg_values = avg_values

    def SetTestAverageValues(self):
        """测试用例 
        func = e^x
        设置精确的单元格积分均值
        """
        for i in range(self.N):
            integrate = np.exp(self.mesh[i + 1]) - np.exp(self.mesh[i])
            self.avg_values[i] = integrate / (self.mesh[i + 1] - self.mesh[i])



