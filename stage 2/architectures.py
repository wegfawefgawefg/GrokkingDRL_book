import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class TinyNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize

        self.l1 = nn.Linear(self.inputSize, self.hiddenSize)
        self.l2 = nn.Linear(self.hiddenSize, self.outputSize)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.softmax(x, dim=0)
        return x

class MedNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize

        self.l1 = nn.Linear(self.inputSize, self.hiddenSize)
        self.lh1 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.lh2 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.lh3 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.lh4 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.l2 = nn.Linear(self.hiddenSize, self.outputSize)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.lh1(x))
        x = F.relu(self.lh2(x))
        x = F.relu(self.lh3(x))
        x = F.relu(self.lh4(x))

        x = self.l2(x)
        x = F.softmax(x, dim=0)
        return x

class MedSkipNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize

        self.l1 = nn.Linear(self.inputSize, self.hiddenSize)
        self.lh1 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.lh2 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.lh3 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.lh4 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.l2 = nn.Linear(self.hiddenSize, self.outputSize)

    def forward(self, x):
        xl1 = F.relu(self.l1(x))
        xh1 = F.relu(self.lh1(xl1))
        xh2 = F.relu(self.lh2(xh1) + 0.1 * xl1)
        xh3 = F.relu(self.lh3(xh2))
        xh4 = F.relu(self.lh4(xh3) + 0.1 * xh2)

        x = self.l2(xh4)
        return x

class DeepNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize

        self.l1 = nn.Linear(self.inputSize, self.hiddenSize)
        self.lh1 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.lh2 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.lh3 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.lh4 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.lh11 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.lh21 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.lh31 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.lh41 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.lh12 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.lh22 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.lh32 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.lh42 = nn.Linear(self.hiddenSize, self.hiddenSize)
        self.l2 = nn.Linear(self.hiddenSize, self.outputSize)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.lh1(x))
        x = F.relu(self.lh2(x))
        x = F.relu(self.lh3(x))
        x = F.relu(self.lh4(x))
        x = F.relu(self.lh11(x))
        x = F.relu(self.lh21(x))
        x = F.relu(self.lh31(x))
        x = F.relu(self.lh41(x))
        x = F.relu(self.lh12(x))
        x = F.relu(self.lh22(x))
        x = F.relu(self.lh32(x))
        x = F.relu(self.lh42(x))
        x = self.l2(x)
        return x