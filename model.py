import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet

class MnistModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b4', include_top=False)
        self.hidden = nn.Linear(1792, 256)
        self.classifier = nn.Linear(256, 26)
        self.counter = nn.Linear(256, 1)
        self.sigmoid = torch.sigmoid

    def forward(self, x):
        x = self.efficientnet(x)
        x = nn.Flatten()(x)
        x = self.hidden(x)
        x1 = self.classifier(x)
        x1 = self.sigmoid(x1)
        x2 = self.counter(x)

        return x1, x2