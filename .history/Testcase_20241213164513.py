# 构建测试用例
import WENO
import numpy as np


def shockwave_fun(x):
    if x <= 0.5:
        return 0
    else:
        return 1

class ShockWave:
    def __init__(self, mesh: WENO.Mesh):
        self.mesh = mesh  # 网格类 不是具体的网格
        self.N = self.mesh.N  # 单元格数
        ...
        
    def getTheoValues(self):
        # 获取网格点上的理论值
        ...
        
    def getAverageValues(self):
        # 获取每个单元格的的平均值
        ...