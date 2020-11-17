import torch

# 1.矩阵的加减乘除
a = torch.rand(2, 2)
b = torch.rand(2, 2)
print(a + b)
print(torch.add(a, b))

print(torch.all(torch.eq(a - b, torch.sub(a, b))))
# */mul 是矩阵的点乘，而不是矩阵的乘法
print(torch.all(torch.eq(a * b, torch.mul(a, b))))
print(torch.all(torch.eq(a / b, torch.divide(a, b))))

# 2.矩阵的乘法有三种方法：Torch.mm（only for 2d）、Torch.matmul、@
a = torch.Tensor([[3, 3], [3, 3]])
b = torch.ones(2, 2)

print(torch.mm(a, b))
print(torch.matmul(a, b))
print(a @ b)

# 这里模拟的是神经网络的一层，输入的是一个 4 * 784 的矩阵，代表了 4 张图片，每一张图片由 28 * 28 = 784，
# 现在想做的是把输入的矩阵压缩为 512，因此需要乘以一个 784 * 512 的矩阵。但是在 pytorch 中矩阵的格式一般是 (channel-out, channel-in）
# 所以写为 512 * 784
b = torch.rand(4, 784)
w = torch.rand(512, 784)
print((b @ w.t()).shape)

a = torch.rand(4, 3, 28, 64)
b = torch.rand(4, 3, 64, 32)
# print(torch.mm(a, b).shape) runtime error，mm 方法只对 2 维的矩阵有效
# 多维矩阵相乘，但是也只是对最后两维进行计算
print(torch.matmul(a, b).shape)

b = torch.rand(4,1, 64, 32)
print(torch.matmul(a, b).shape)

b = torch.rand(4, 64, 32)
# print(torch.matmul(a, b).shape) runtime error

# 3.矩阵的幂
a = torch.full([2, 2], 3, dtype=torch.float)
print(a.pow(2))
print(a ** 2)
# 开方
print((a ** 2).sqrt())
# 开方，并且取倒数
print((a ** 2).rsqrt())
print((a ** 2) ** 0.5)

# 4.近似值
a = torch.tensor(3.14)
# floor 向下取整，ceil 向上取整，trunc 直接截断小数部分，frac 直接截断整数部分
print("floor: {0}, ceil:{1}, trunc:{2}, frac:{3}".format(a.floor(), a.ceil(), a.trunc(), a.frac()))
a = torch.tensor(3.4999)
print(a.round())
a = torch.tensor(3.5)
print(a.round())

# 5.clamp 方法
grad = torch.rand(2, 3) * 15
print(grad)
print(grad.max())
# grad 矩阵中的中位数
print(grad.median())
# 将 input 中的元素限制在 [min, 无穷] 之间，小于 min 的转变为 min
print(grad.clamp(10))

grad = torch.tensor([[-1, 20], [7, 18]])
# 将 input 中的元素限制在 [min,max] 范围内，小于 min 的变为 min，大于 max 的变为 max
print(grad.clamp(0, 10))

