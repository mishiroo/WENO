import WENO
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
    mesh = WENO.Mesh(0, 5, n)  # 实例化的Mesh mesh.mesh访问网格本身
    mesh.SetTestAverageValues()
    theo_values = np.exp(mesh.mesh)
    weno = WENO.WENO1D(mesh=mesh.mesh, avg_values=mesh.avg_values, k=5)

    # 模板为5的计算结果
    weno5_solution = weno.StencilReconstruct_all()
    # weno.Reconstruct()
    # 处理边界条件
    # weno.BoundaryTreatment(theo_values)
    weno5_solution = weno._solution

    # 统计误差
    linf_err_list.append(EA.LinfError(theo_values, weno5_solution))
    l2_err_list.append(EA.L2Error(theo_values, weno5_solution))
    solution_list.append(weno5_solution)
    # 绘图
    EA.Draw(mesh.mesh, theo_values, weno5_solution)

# np.save("k=5加权结果.npy", solution_list[0])

EA.SaveError("WENO5Error.xlsx", NList, l2_err_list, linf_err_list)
