import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class MnistModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b5')
        # self.efficientnet = EfficientNet.from_name('efficientnet-b1')
        self.classifier = nn.Linear(1000, 26)

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.classifier(x)

        return x
