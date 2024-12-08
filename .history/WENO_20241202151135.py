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
        assert np.size(mesh) - 1 == np.size(avg_values), "len(mesh) == len(avg_values) + 1"
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
            self.weights_d = [3 / 10, 3 / 5, 1 / 10]
        else:
            raise ValueError("order must be 1, 2 or 3")  # 其他阶的系数论文没给，先不写

        self.C = np.array(
            [
                [1 / 3, 5 / 6, -1 / 6],
                [-1 / 6, 5 / 6, 1 / 3],
                [1 / 3, -7 / 6, 11 / 6],
                [11 / 6, -7 / 6, 1 / 3],
            ]
        )

    def reconstruct(self, index: int) -> List[float]:
        # index: 0,1,2,...,N
        # 获取操作stencil
        if index < 0 or index > self.N:
            raise ValueError("index out of range!")
        elif self.k <= index <= self.N - self.k + 1:
            self.stencil = self.avg_values[index - self.k + 1 : index + self.k]
            reconstruct_values = np.zeros(self.k)  # reconstruct values
            for i in range(self.k):
                reconstruct_values[i] = np.inner(
                    self.C[i], self.stencil[i : i + self.k]
                )
        # 对于没有k个模板的边界位置 左端点
        elif index < self.k:
            v0 = np.inner(self.C[-1], self.avg_values[: self.k])   
            return [v0, v0, v0]
        # 对于没有k个模板的边界位置 右端点
        elif index > self.N - self.k + 1:
            vN =np.inner(self.C[2], self.avg_values[self.N - self.k : self.N]) 
            return [vN, vN, vN]
        return reconstruct_values


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
        
    
    def Testcase0(self):
        """测试用例 返回理论值
        func = e^x
        """
        for i in range(self.N):
            integrate = np.exp(self.mesh[i+1]) - np.exp(self.mesh[i])
            self.avg_values[i] = integrate / (self.mesh[i+1] - self.mesh[i])
        return np.exp(self.mesh)


# 测试
mesh = Mesh(0, 1, 100)
theo_values = mesh.Testcase0()
weno = WENO1D(mesh=mesh, avg_values=mesh.avg_values)
print(weno.reconstruct(0))
