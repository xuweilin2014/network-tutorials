import torch

# 1.norm，这里的 norm 方法不是正则化的意思，而是求出 tensor 所给的矩阵范数或者向量范数的意思
# 若 x = [x1, x2, x3, ..., xp]T，那么 x 的 p 范数为：
# ||x||p = (|x1|^p + |x2|^p + |x3|^p + ... + |xn|^p) ^ (1/p)
a = torch.full([8], 1, dtype=torch.float)
b = a.view(2, 4)
c = a.view(2, 2, 2)

print(b)
print(c)
print("a.norm(1):{0}, b.norm(1):{1}, c.norm(1):{2}".format(a.norm(1), b.norm(1), c.norm(1)))
print("a.norm(2):{0}, b.norm(2):{1}, c.norm(2):{2}".format(a.norm(2), b.norm(2), c.norm(2)))

print(b.norm(1, dim=1))
print(b.norm(2, dim=1))
print(c.norm(1, dim=0))
print(c.norm(2, dim=0))

# 2. argmax, argmin
a = torch.arange(8).view(2, 4).float()
print(a)
# 求出 a 矩阵中的最小值，最大值，平均值以及累乘
print(a.min(), a.max(), a.mean(), a.prod())
print(a.sum())
# argmax 返回矩阵 a 中最大值的索引，argmin 返回矩阵 a 中最小值的索引
# 如果不指定维度 dim 参数，返回的索引是将矩阵的维度变为 1 之后的索引
print(a.argmax(), a.argmin())

a = torch.randn(3, 6)
print(a)
print(a.argmax())
print(a.argmax(dim=1))

print(a.max(dim=1, keepdim=True))
print(a.argmax(dim=1, keepdim=True))

# 3.torch.eq，> >= < <= != ==
print(a > 0)
# greater than
print(torch.gt(a, 0))
print(a != 0)

a = torch.ones(2, 3)
b = torch.randn(2, 3)
print(torch.eq(a, b))
print(torch.eq(a, a))
print(torch.equal(a, a))


