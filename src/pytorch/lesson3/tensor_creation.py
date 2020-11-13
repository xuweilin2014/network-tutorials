import numpy as np
import torch

# 创建 pytorch 中 tensor 的几种方式

# 1.第一种创建 tensor 的方式，使用 numpy 先创建，然后使用 torch 进行导入
a = np.array([2, 3.3])
print(a)
a = torch.from_numpy(a)
print(a)

a = np.ones([2, 3])
print(a)
a = torch.from_numpy(a)
print(a)

# 2.第二种创建 tensor 的方式，适用于数据量不大的情况，直接使用 tensor 来创建
# torch.tensor 接受的是真正的数据，也就是作为参数的是什么数据，最后输出的就是什么数据
# torch.Tensor/torch.FloatTensor 既可以接受真正的数据，也可以接受 shape 的值
a = torch.tensor([2, 3.2])  # 真正的数据
print(a)
a = torch.FloatTensor([2, 3.2])  # 真正的数据
print(a)
a = torch.FloatTensor(2, 3)  # 生成一个 2 * 3 的矩阵，矩阵中的数据是未初始化的随即值
print(a)
a = torch.tensor([[2, 3.2], [1, 22.3]])
print(a)

# 3.第三种创建 tensor 的方式，创建一个未初始化的区域，这个区域中的值都是随机值
a = torch.empty(2, 3)
print(a)
a = torch.Tensor(2, 3)
print(a)
a = torch.FloatTensor(2, 3)
print(a)
a = torch.IntTensor(2, 3)
print(a)

# torch.Tensor 初始化的类型默认一般是 torch.FloatTensor
print(torch.Tensor(1, 1).type())
print(torch.Tensor([1, 1]).type())

# 4.第四种创建 tensor 的方式，创建一个未初始化的区域，和第三种不同，使用 rand/randn 函数
# 创建一个大小为 2 * 3 的矩阵，但是这个矩阵中的值为 0 ~ 1
a = torch.rand(2, 3)
print(a)
# 创建一个 2 * 3 的矩阵，矩阵中的值符合正态分布 N(0, 1)
a = torch.randn(2, 3)
print(a)
a = torch.rand(4, 5)
# 创建一个矩阵，这个矩阵的大小和传入的参数 a 的大小一样，同时值为 0 ~ 1
a = torch.rand_like(a)
print(a)
# 创建一个 3 * 3 的矩阵，这个矩阵的值的区间为 1 ~ 10
a = torch.randint(1, 10, [3,3])
print(a)

# 5.创建一个 2 * 3 的矩阵，矩阵中的值都为 78
a = torch.full([2, 3], 78)
print(a)
# 创建一个 0 维的向量，或者说创建一个标量，值为 78
a = torch.full([], 78)
print(a)
# 创建一个数组，数组的大小为 2，数组中的每个值为 78
a = torch.full([2], 78)
print(a)
a = torch.full([1], 78)
print(a)