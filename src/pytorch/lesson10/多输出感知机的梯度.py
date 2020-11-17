import torch
from torch.nn import functional as F

x = torch.randn(1, 10)
# 因为此感知机有多个输出，所以 w 的大小为 2 * 10
w = torch.randn(2, 10)
w.requires_grad_()
o = torch.sigmoid(torch.matmul(x, w.t()))
print(o.shape)

loss = F.mse_loss(torch.ones(1, 2), o)
print(loss)

loss.backward()
print(w.grad)