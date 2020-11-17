import torch
from torch.nn import functional as F

# 1.激活函数 sigmoid
a = torch.linspace(-100, 100, 10)
print(a)
print(torch.sigmoid(a))

# 2.激活函数 tanh
a = torch.linspace(-1, 1, 10)
print(a)
print(torch.tanh(a))

# 3.  relu 激活函数
a = torch.linspace(-1, 1, 10)
print(a)
print(torch.relu(a))
print(F.relu(a))

# 4.均方差（Mean Squared Error）的 loss 函数梯度
# MSE = Sum((y[i] - prediction[i])^2) = Sum((y[i] - (w * x[i] + b)) ^ 2)
# torch.autograd.grad 可以自动进行求导
x = torch.ones(1, dtype=torch.float)
print(x)
w = torch.full([1], 2).float()
print(w)
mse = F.mse_loss(torch.ones(1), x * w)
print(mse)

# 4.1 torch.autograd.grad(loss, [w1, w2, w3,...]) 来计算梯度，也就是自动进行求导
# print(torch.autograd.grad(mse, [w]))
# RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
w.requires_grad_()
print(w)
mse = F.mse_loss(torch.ones(1), x * w)
print(torch.autograd.grad(mse, [w]))

# 4.2 loss.backward() 来计算梯度，月和可以用来进行求导
mse = F.mse_loss(torch.ones(1), x * w)
mse.backward()
print(w.grad)

# 5.softmax 激活函数
# y1 = 2.0 y2 = 1.0 y3 = 0.1
# softmax(yi) = (e ^ yi) / Sum(e ^ yi)
# softmax(y1) = 0.7 softmax(y2) = 0.2 softmax(y3) = 0.1
a = torch.rand(3)
print(a)
a.requires_grad_()
p = F.softmax(a, dim=0)
print(p)
# 使用 backward 来计算梯度，注意 loss.backward 这里的 loss 必须是一个 1 * 1 的 tensor
p[0].backward(retain_graph=True)
print(a.grad)
# 使用 torch.autograd.grad(loss, [w1,w2,w3,...]) 来计算梯度，这里的 loss 的值也应该是一个 1 * 1 的量
print(torch.autograd.grad(p[0], a))
