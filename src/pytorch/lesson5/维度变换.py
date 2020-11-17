import torch

# pytorch 中 view 和 reshape 这两个方法是一样的
a = torch.rand(4, 1, 28, 28)
print(a.shape)
print(a.view(4, 28, 28))
print(a.view(4, 28, 28).shape)
print(a.view(4 * 28, 28).shape)
print(a.view(4 * 1, 28, 28).shape)

# unsqueeze 方法是展开，也就是增加维度，方法的参数值可以为正值，也可以为负值，当为正值的时候，
# 表示是在之前插入维度，如果为负值，表示在之后插入维度
print(a.shape)
print(a.unsqueeze(0).shape)
print(a.unsqueeze(-1).shape)
print(a.unsqueeze(4).shape)
print(a.unsqueeze(-4).shape)
print(a.unsqueeze(-5).shape)
# error occurs
# print(a.unsqueeze(5).shape)

a = torch.tensor([1.2, 2.3])
print(a.unsqueeze(-1))
print(a.unsqueeze(0))

# squeeze 方法是挤压，缩小维度，和前面的 unsqueeze 方法一样也是接收一个 index 作为参数
b = torch.rand(1, 32, 1, 1)
print(b.shape)
print(b.squeeze().shape)
print(b.squeeze(0).shape)
print(b.squeeze(-1).shape)
print(b.squeeze(1).shape)
print(b.squeeze(-4).shape)

# torch.Tensor 有两个实例方法可以用来扩展某维的数据尺寸，分别是 repeat 和 expand
# expand 返回当前张量在某维扩展更大后的张量。扩展（expand）张量不会分配新的内存，只是在存在的张量上创建一个新的视图（view），一个大小（size）等于1的维度扩展到更大的尺寸
x = torch.tensor([1, 2, 3])
print(x.expand(2, 3))
x = torch.randn(2, 1, 1, 4)
# -1 表示该维度的尺寸保持不变，旧维度变为新维度，旧的维度必须为 1
print(x.expand(-1, 2, 4, -1).shape)

# repeat 方法沿着特定的维度重复这个张量，和expand()不同的是，这个函数拷贝张量的数据。所以，repeat 方法的参数不是新的 shape，而是旧的 shape 在这个维度需要重复的次数
x = torch.tensor([1, 2, 3])
# x 的 shape 为 1 * 3，而下面的 repeat 方法的参数的意思是第一个维度数据重复 3 次，第二个维度的数据重复 2 次
print(x.repeat(3, 2))
x = torch.randn(2, 3, 4)
print(x.repeat(2, 1, 3).shape)

# transpose 方法
a = torch.rand(4, 3, 32, 32)
# 没有转置回来，a1 和 a 的 shape 虽然相等，但是数据发生了变化
a1 = a.transpose(1, 3).contiguous().view(4, 3 * 32 * 32).view(4, 3, 32, 32)
# 先进行转置，再转置回来，a2 和 a 的 shape 相等，数据也相等
a2 = a.transpose(1, 3).contiguous().view(4, 3 * 32 * 32).view(4, 32, 32, 3).transpose(1 , 3)
print("a1 shape:{0}, a2 shape:{1}".format(a1.shape, a2.shape))
print(torch.all(torch.eq(a, a1)))
print(torch.all(torch.eq(a, a2)))

# permute 方法将 tensor 的维度换位，参数是一系列的整数，代表原来张量的维度。比如三维就有0，1，2这些dimension
a = torch.rand(4, 3, 28, 28)
print(a.transpose(1, 3).shape)
b = torch.rand(4, 3, 28, 32)
print(b.transpose(1, 3).shape)
print(b.transpose(1, 3).transpose(1, 2).shape)
print(b.permute(0, 2, 3, 1).shape)