import torch
import torch.nn as nn
import timm
import neptune
import numpy as np
import torch_optimizer as optim
from efficientnet_pytorch import EfficientNet
from pytorch_lightning import LightningModule
from sklearn.metrics import roc_auc_score
from pytorch_lightning.metrics import Accuracy

class MnistModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b5')
        self.classifier = nn.Linear(1000, 26)

    def forward(self, x):
        x = self.efficientnet(x)
        x = self.classifier(x)

        return x


# class MnistModel(LightningModule):
#     def __init__(self, model: nn.Module = None):
#         super().__init__()
#         self.model = model
#         self.criterion = nn.MultiLabelSoftMarginLoss()
#         self.metric = Accuracy(threshold=0.5)

#     def forward(self, x):
#         x = self.model(x)
#         return x

#     def training_step(self, batch, batch_nb):
#         x, y = batch
#         y_hat = self(x)

#         loss = self.criterion(y_hat, y)
#         acc = self.metric(y_hat, y)

#         neptune.log_metric('train_loss', loss)
#         neptune.log_metric('train_accuracy', acc)

#         return loss     

#     def validation_step(self, batch, batch_nb):
#         x, y = batch
#         y_hat = self(x)

#         loss = self.criterion(y_hat, y)
#         acc = self.metric(y_hat, y)

#         return {'loss': loss, 'acc': acc}


#     def validation_epoch_end(self, outputs):
#         avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
#         avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

#         self.log('val_loss', avg_loss, prog_bar=True)
#         self.log('val_acc', avg_acc, prog_bar=True)

#         neptune.log_metric('val_loss', avg_loss)
#         neptune.log_metric('val_acc', avg_acc)   


#     def configure_optimizers(self):
#         # optimizer = optim.Adam(model.parameters(), lr=1e-3)
#         # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#         optimizer = optim.RAdam(
#             model.parameters(),
#             lr= 1e-3,
#             betas=(0.9, 0.999),
#             eps=1e-8,
#             weight_decay=0,
#         )
#         optimizer = optim.Lookahead(optimizer, k=5, alpha=0.5)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=10, eta_min=0)

#         return [optimizer], [scheduler]


