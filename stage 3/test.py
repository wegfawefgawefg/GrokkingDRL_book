import torch

a = torch.tensor([[1, 2],[3, 4],[5, 6],[7, 8],[9, 0],])
b = torch.tensor([0, 1, 0, 0, 1])
k = torch.tensor([11, 12, 13, 14, 15]).view(5, 1)
c = b.view(5, 1)
d = a.scatter_(1, c, k)
print(a)
print(b)
print(c)
print(d)