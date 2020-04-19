import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

a = torch.tensor([1,2,3], dtype=torch.float32)
b = F.softmax(a, dim=0)
print(b)