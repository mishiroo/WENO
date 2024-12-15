import WENO
import ErrorAnalysis as EA
import numpy as np

# 计算误差阶
# N: Num of cells
NList = [100, 200, 400, 800, 1600]

for n in NList:
    print(f"num of cells = {n}...")
    mesh = WENO.Mesh(0, 1, n)
    mesh.SetTestAverageValues()
    theo_values = np.exp(mesh.mesh)
    weno = WENO.WENO1D(mesh=mesh.mesh, avg_values=mesh.avg_values)
    
    weno5_solution = weno.Reconstruct()
    # 统计误差
    weno.linf_err_list.append(EA.LinfError(theo_values, weno5_solution))
    weno.l2_err_list.append(EA.L2Error(theo_values, weno5_solution))
    weno.solution_list.append(weno5_solution)
    # 绘图
    EA.Draw(mesh.mesh, theo_values, weno5_solution)


EA.SaveError("WENO5Error.xlsx", NList, L2ErrorList, LinfErrorList)



