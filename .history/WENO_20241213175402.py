import tqdm
import numpy as np
from typing import List


# 1D dimension
class WENO1D:

    def __init__(self, mesh: np.ndarray, avg_values: np.ndarray, k: int = 5) -> None:
        """初始化WENO类

        Args:
            mesh (np.ndarray): 网格剖分[lb,...ub]
            avg_values (np.ndarray): 网格剖分[lb,...ub]
            k (int, optional): 模板长度. Defaults to 3.
            Supported k=1, 2, 3, 5
        """
        assert np.size(mesh) - 1 == np.size(
            avg_values
        ), "len(mesh) == len(avg_values) + 1"
        self.avg_values = avg_values
        self.mesh = mesh

        self.order = 2 * k - 1
        self.lb = mesh[0]
        self.ub = mesh[-1]
        self.N = len(mesh) - 1  # Num of cells
        self.k = k  # Num of stencils

        self._solution = np.zeros(self.N + 1)  # 存储WENO重构结果

        # 查表得到模板组合线性权
        if self.order == 1:
            self.weights_d = [1]
        elif self.order == 3:
            self.weights_d = [2 / 3, 1 / 3]
        elif self.order == 5:
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
        elif self.order == 9:
            # 直接得到五阶模板 加权可以得到九阶模板
            self.weights_d = np.array([0, 0, 1, 0, 0])  # 书上没有对应权重 取中间的权重
            self.C = np.array(
                [
                    [1 / 5, -21 / 20, 137 / 60, -163 / 60, 137 / 60],
                    [-1 / 20, 17 / 60, -43 / 60, 77 / 60, 1 / 5],
                    [1 / 30, -13 / 60, 47 / 60, 9 / 20, -1 / 20],
                    [-1 / 20, 9 / 20, 47 / 60, -13 / 60, 1 / 30],
                    [1 / 5, 77 / 60, -43 / 60, 17 / 60, -1 / 20],
                    [137 / 60, -163 / 60, 137 / 60, -21 / 20, 1 / 5],
                ]
            )
        else:
            raise ValueError(
                "k must be 1, 2 or 3, 5(测试五阶算法用)"
            )  # 其他阶的系数论文没给，先不写

        self.PrintInfo()

    def PrintInfo(self):
        print(
            f"lb = {self.lb}, ub = {self.ub}, Stencil length = {self.k}, WENO order = {self.order} "
        )
        print(f"Num of cells = {self.N}, Num of Nodes = {self.N + 1} \n")

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
                    self.avg_values[stencil_index + 1 : stencil_index + self.k + 1],
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
                        self.avg_values[index]
                        - 2 * self.avg_values[index + 1]
                        + self.avg_values[index + 2]
                    )
                    ** 2
                )
                +1 / 4 * (
                    3 * self.avg_values[index]
                    - 4 * self.avg_values[index + 1]
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
            return [beta0, beta1, beta2][::-1]

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

    def NodeReconstruct(self, index: int, eps=1e-6) -> float:
        """WENO加权后的重构 精度为2k-1阶 针对节点进行重构

            返回指定索引的WENO重构结果

        Args:
            index (int): 索引
            eps (_type_, optional): eps参数. Defaults to 1e-6.

        Returns:
            float: 插值结果
        """
        weights = self.FormWeights(index, eps=eps)
        print(weights)
        return np.inner(weights, self.StencilReconstruct(index))

    def Reconstruct(self, eps=1e-6):
        """WENO加权后的重构 精度为2k-1阶 针对区间进行重构

            返回区间上每个节点的WENO重构结果

        Args:
            eps (_type_, optional): eps参数. Defaults to 1e-6.
        """
        for i in tqdm.tqdm(range(self.N + 1)):
            self._solution[i] = self.NodeReconstruct(i, eps=eps)
        return self._solution

    def StencilReconstruct_all(self) -> np.ndarray:
        """精度为k阶 调用StencilReconstruct 针对区间进行重构

            返回区间所有节点使用模板重构结果(收敛阶为模板长度)

        Returns:
            np.ndarray: 重构结果
        """
        for i in range(self.N + 1):
            self._solution[i] = np.inner(self.weights_d, self.StencilReconstruct(i))
        return self._solution

    def BoundaryTreatment(self, theo_values: np.ndarray) -> None:
        """边界处理: 把遇到的不完整模板计算结果全部用精确值代替

        Args:
            theo_values (np.ndarray): 理论值
        """
        for i in range(self.k):
            # 前k个点
            self._solution[i] = theo_values[i]
            # 后k个点
            self._solution[self.N - i] = theo_values[self.N - i]


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
