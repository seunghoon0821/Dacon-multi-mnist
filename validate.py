import torch
from torch.utils.data import DataLoader
from torch import nn
from dataset import MnistDataset
from preprocess import transforms_train, transforms_test
from model import MnistModel
from torchinfo import summary
import torch.optim as optim
import neptune

def validate(test_loader, model, criterion, epoch, device):
    model.eval() # prep model for *evaluation*

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)

            outputs = outputs > 0.5
            acc = (outputs == targets).float().mean()
            return loss, acc
