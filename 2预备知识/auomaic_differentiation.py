import torch

x = torch.arange(4.0)
x.requires_grad_(True)
print(x.grad)

y = 2 * torch.dot(x, x)
print(y)

y.backward()
print(x.grad)
