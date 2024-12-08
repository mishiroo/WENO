# 测试
mesh = Mesh(0, 1, 101)
theo_values = mesh.Testcase0()
weno = WENO1D(mesh=mesh.mesh, avg_values=mesh.avg_values)
# 计算误差
for i in range(weno.N):
    print(weno.StencilReconstruct(i) - theo_values[i])
