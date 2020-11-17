import torch

# 合并，主要有两个方法：cat 和 stack
a = torch.rand(4, 32, 8)
b = torch.rand(5, 32, 8)
# 在指定维度 dim 进行拼接操作，a 和 b 的维度的数量必须一致，这里均为 3，并且除了指定的 dim 可以不一样之外，其余的维度必须保持一致
print(torch.cat([a, b], dim=0).shape)

a1 = torch.rand(4, 3, 32, 32)
a2 = torch.rand(5, 3, 32, 32)
print(torch.cat([a1, a2], dim=0).shape)

a2 = torch.rand(4, 1, 32, 32)
# print(torch.cat([a1, a2], dim=0).shape) runtime error
print(torch.cat([a1, a2], dim=1).shape)

a1 = torch.rand(4, 3, 16, 32)
a2 = torch.rand(4, 3, 16, 32)
print(torch.cat([a1, a2], dim=2).shape)

# 对于torch.stack来说，会先将原始数据维度扩展一维，然后再按照维度进行拼接，具体拼接操作同torch.cat类似
print(torch.stack([a1, a2], dim=2).shape)
a1 = torch.rand(32, 8)
a2 = torch.rand(32, 8)
print(torch.stack([a1, a2], dim=0).shape)

a1 = torch.rand([30, 8])
a2 = torch.rand([32, 8])
# print(torch.stack([a1, a2], dim=0)) runtime error, stack 方法也要求 dim 维度以及之后的维度数保持一致
print(torch.cat([a1, a2], dim=0).shape)

# 分割：主要是 split 方法
a1 = torch.rand([3, 32, 8])
a2 = torch.rand([5, 32, 8])
a3 = torch.cat([a1, a2], dim=0)
# a3 shape: [8, 32, 8]
print(a3.shape)
# 1 + 3 + 4 必须等于 8
aa, bb, cc = a3.split([1, 3, 4], dim=0)
print("aa shape:{0}, bb shape:{1}, cc shape:{2}".format(aa.shape, bb.shape, cc.shape))
aa, bb, cc, dd = a3.split(2, dim=0)
print(aa.shape, bb.shape, cc.shape, dd.shape)