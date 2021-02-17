import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet

class MnistModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b1')
        self.classifier = nn.Linear(1000, 26)
        self.counter = nn.Linear(1000, 1)
        self.sigmoid = torch.sigmoid

    def forward(self, x):
        x = self.efficientnet(x)
        x1 = self.classifier(x)
        x1 = self.sigmoid(x1)
        x2 = self.counter(x)

        return x1, x2