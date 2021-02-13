import torch
from torch.utils.data import DataLoader
from torch import nn
from dataset import MnistDataset
from preprocess import transforms_train, transforms_test
import numpy as np

def validate(test_loader, model, criterion, epoch, device):
    model.eval() # prep model for *evaluation*
    val_loss = 0 
    val_acc = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)

            outputs = outputs > 0.5
            val_acc += (outputs == targets).float().mean()*images.size(0)
            val_loss += loss.item()*images.size(0)
    
    val_loss = val_loss/len(test_loader.dataset)
    val_acc = val_acc/len(test_loader.dataset)

    return val_loss, val_acc
