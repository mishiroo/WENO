import WENO
import ErrorAnalysis as EA
import numpy as np
import pandas as pd


# 计算误差阶
# N: Num of cells
NList = [100, 200, 400, 800, 1600]
L2ErrorList = []
LinfErrorList = []
Solutions = []

for n in NList:
    mesh = WENO.Mesh(0, 1, n)
    mesh.SetTestAverageValues()
    theo_values = np.exp(mesh.mesh)
    weno = WENO.WENO1D(mesh=mesh.mesh, avg_values=mesh.avg_values)
    weno5_solution = np.zeros_like(mesh.mesh)
    for i in range(weno.N):
        weno5_solution[i] = weno.WENOReconstruct(i)
    LinfErrorList.append(np.max(np.abs(weno5_solution - theo_values)))
    L2ErrorList.append(np.sqrt(np.sum((weno5_solution - theo_values) ** 2)))
    Solutions.append(weno5_solution)

ac_orders_l2 = EA.AccuracyOrders(NList, L2ErrorList)
ac_orders_linf = EA.AccuracyOrders(NList, LinfErrorList)
# 收敛阶的长度正好比NList少1，需要补一个None


# 保存计算结果
data = {
    "N": NList,
    "L2Error": L2ErrorList,
    "L2Order": ac_orders_l2,
    "LInfError": LinfErrorList,
    "LInfOrder": ac_orders_linf,
}
df = pd.DataFrame(data)
df.to_excel("output.xlsx", index=False)