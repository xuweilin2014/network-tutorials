import torch

a = torch.rand(4, 3, 28, 28)
print(a[0].shape)
print(a[0, 0].shape)
print(a[0, 0, 2, 4].shape)
print(a[0, 0, 2, 4])
print()

print(a.shape)
print(a[:2].shape)
print(a[:2, :1, :, :].shape)
print(a[:2, 1:, :, :].shape)
print()

print(a[:,:, 0:28:2, 0:28:2].shape)
print(a[:, :, ::2, ::2].shape)
print()

print(a.shape)
print(a.index_select(0, torch.tensor([0, 2])).shape)
print(a.index_select(1, torch.tensor([1, 2])).shape)
print(a.index_select(2, torch.arange(28)).shape)
print(a.index_select(2, torch.arange(8)).shape)
print()

print(a.shape)
print(a[...].shape)
print(a[0, ...].shape)
print(a[:, 1, ...].shape)
print(a[..., :2].shape)