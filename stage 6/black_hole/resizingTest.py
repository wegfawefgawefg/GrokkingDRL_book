import torch

batchSize = 4
params = 2
numActions = 3
a = torch.arange(24).view(batchSize, params, numActions)
print(a)

# a = torch.tensor([1,2,3,4])
# b = a.unsqueeze()
# print(b)

b, c = a[:][:, 0], a[:][:, 1]
print(b)
print(c)