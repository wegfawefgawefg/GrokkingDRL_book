import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

class TwoNet(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenShape):
        super().__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenShape = hiddenShape

        self.input_layer = nn.Linear(self.inputSize, self.hiddenShape)
        self.output_layer = nn.Linear(self.hiddenShape, self.outputSize)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        x = self.output_layer(x)
        # x = F.softmax(x, dim=0)
        return x


class FlexNet(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenShape=(32, 32)):
        super().__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenShape = hiddenShape

        self.input_layer = nn.Linear(self.inputSize, hiddenShape[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hiddenShape)-1):   #   skip  (already made first layer)
            hidden_layer = nn.Linear(hiddenShape[i], hiddenShape[i+1])
            self.hidden_layers.append(hidden_layer)

        self.output_layer = nn.Linear(self.hiddenShape[-1], self.outputSize)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = F.relu(x)
        x = self.output_layer(x)
        return x
