import torch

a = torch.tensor([3.4, 50.0, 234.5])
b = torch.exp(a)
print(b)
print(b.tolist())