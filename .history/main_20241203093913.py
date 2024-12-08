import WENO
import ErrorAnalysis as EA
import numpy as np
import tqdm

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
    for i in tqdm.tqdm(range(weno.N+1)):
        weno5_solution[i] = weno.WENOReconstruct(i)
    LinfErrorList.append(EA.LinfError(theo_values, weno5_solution))
    L2ErrorList.append(EA.L2Error(theo_values, weno5_solution))
    Solutions.append(weno5_solution)


EA.SaveError("WENO5Error.txt", NList, L2ErrorList, LinfErrorList)



