import torch
from torch.nn import functional as F

# x 表示神经网络的输入，一共有 10 个输入，[x0, x1, x2, ..., x9]
x = torch.randn(1, 10)
# w 表示神经网络第一层的权重，一共有 10 个参数，[w0, w1, w2, ..., w9]
w = torch.randn(1, 10, requires_grad=True)
# x @ w 之后再通过 sigmod 激活函数，也就是通过这个神经网络的前向传播得到的一个预测值
o = torch.sigmoid(torch.matmul(x, w.t()))
print(o.shape)

# 计算神经网络的预测值 o 和实际值的均方差
loss = F.mse_loss(torch.ones(1, 1), o)
print(loss.shape)

# 反向传播得到梯度
loss.backward()
print(w.grad)
