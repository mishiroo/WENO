import WENO
import TestCase
import ErrorAnalysis as EA
import numpy as np

# 计算误差阶
# N: Num of cells 网格密度太大 机器误差会影响收敛
NList = [128, 256, 512, 1024]
# 存储计算结果
l2_err_list = []
linf_err_list = []
solution_list = []

for n in NList:
    mesh = WENO.Mesh(0, 1, n)  # 实例化的Mesh mesh.mesh访问网格本身
    test_case = TestCase.TestCase(mesh)
    # 用户只需要实现需要的函数即可
    theo_values = test_case.getTheoValues(TestCase.shockwave_fun)
    avg_values = test_case.shockwave_avg()
    
    # 实例化
    weno = WENO.WENO1D(mesh=mesh.mesh, avg_values=avg_values, k=3)
    weno.Reconstruct()  # WENO重构
    weno.BoundaryTreatment(theo_values)

    # 统计误差
    linf_err_list.append(EA.LinfError(theo_values, weno._solution))
    l2_err_list.append(EA.L2Error(theo_values, weno._solution))
    solution_list.append(weno._solution)
    # 绘图
    EA.Draw(mesh.mesh, theo_values, weno._solution)

EA.SaveError("WENO5Error-shock.xlsx", NList, l2_err_list, linf_err_list)