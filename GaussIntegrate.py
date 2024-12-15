import numpy as np


def gauss_integrate(func, interval, n=5):
    """
    使用高斯-勒让德积分方法计算一维积分。

    参数:
    func: 被积分的函数
    a, b: 积分区间 [a, b]
    n: 积分的阶数，默认为5阶精度

    返回:
    积分结果
    """
    a, b = interval
    # 高斯-勒让德积分节点和权重，来自于预计算表
    nodes, weights = np.polynomial.legendre.leggauss(n)

    # 将节点从[-1, 1]映射到[a, b]区间
    transformed_nodes = 0.5 * (b - a) * nodes + 0.5 * (a + b)

    # 计算高斯-勒让德积分结果
    integral = 0.5 * (b - a) * np.sum(weights * func(transformed_nodes))

    return integral

def gauss_point(x_bound: list[float]) -> list[float]:
    """获取高斯点坐标list

    Args:
        x_bound (list[float]): 积分区间

    Returns:
        list[float]: 坐标点
    """
    a, b = x_bound
    # 映射后的高斯点
    c_ = 2 / 7 * np.sqrt(6 / 5)
    gpt = np.array([
        -np.sqrt(3 / 7 + c_),
        -np.sqrt(3 / 7 - c_),
        np.sqrt(3 / 7 - c_),
        np.sqrt(3 / 7 + c_),
    ])
    return ((b - a) * gpt + b + a) / 2

# 设计这些函数的作用是因为如果给出的句柄存在if-else判断，计算高斯点函数值时，会报错
# 所以现在要把计算函数值和组装分开
def gauss_point_value(f, gpt: list, x_bound: list[float]) -> list[float]:
    """获取高斯点函数值

    Args:
        f (function): 函数
        gpt (list): 映射后的高斯点坐标
        x bound: 积分区间

    Returns:
        list: 函数值
    """
    a, b = x_bound
    return [f(x) for x in gpt]


def gauss_integrate_ls(f_list, x_bound: list[float]) -> float:
    """4个高斯点的高斯积分 精度为7阶

    Args:
        f_list : 函数求值后的列表
        x_bound (list[float]): 积分区域

    Returns:
        float: 积分结果
    """
    a, b = x_bound
    # 权重
    w1 = (18 + np.sqrt(30)) / 36
    w2 = (18 - np.sqrt(30)) / 36
    A = [w2, w1, w1, w2]

    # 转为标准区间
    I = 0
    J = (b - a) / 2
    for i in range(4):
        I += A[i] * f_list[i] * J
    return I


# 测试用例
# if __name__ == "__main__":
#     def f(x):
#         return x

#     bd = [0, 2]
#     gpt = gauss_point(bd)
#     f_list = gauss_point_value(f, gpt, bd)
#     print(2)
#     print(gauss_integrate(f, bd))
#     print(gauss_integrate_ls(f_list, bd))
