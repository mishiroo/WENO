import WENO
import ErrorAnalysis as EA
import numpy as np

# 计算误差阶
# N: Num of cells
NList = [100, 200, 400, 800, 1600]
# 存储计算结果
l2_err_list = []
linf_err_list = []
solution_list = []

for n in NList:
    print(f"num of cells = {n}...")
    mesh = WENO.Mesh(0, 1, n)  # 实例化的Mesh mesh.mesh访问网格本身
    mesh.SetTestAverageValues()
    theo_values = np.exp(mesh.mesh)
    weno = WENO.WENO1D(mesh=mesh.mesh, avg_values=mesh.avg_values, order=9)
    
    # weno5_solution = weno.Reconstruct()
    # 模板为5的计算结果
    weno5_solution = weno.StencilReconstruct_all()
    # 统计误差
    linf_err_list.append(EA.LinfError(theo_values, weno5_solution))
    l2_err_list.append(EA.L2Error(theo_values, weno5_solution))
    solution_list.append(weno5_solution)
    # 绘图
    EA.Draw(mesh.mesh, theo_values, weno5_solution)


EA.SaveError("WENO5Error.xlsx", NList, l2_err_list, linf_err_list)



