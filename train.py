import torch
import torch.optim as optim
import neptune

from torch.utils.data import DataLoader
from torch import nn
from dataset import MnistDataset
from preprocess import transforms_train, transforms_test
from model import MnistModel
from torchinfo import summary
from validate import validate
from sklearn.model_selection import train_test_split

# Init Neptune
neptune.init(project_qualified_name='dhdroid/Dacon-MNIST',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZWM3ZDFmYjAtM2FlNS00YzUzLThjYTgtZjU3ZmM1MzJhOWQ4In0=',
             )
neptune.create_experiment()

# cuda cache 초기화
torch.cuda.empty_cache()

# Prepare Data
full_dataset = MnistDataset('data/train', 'data/dirty_mnist_2nd_answer.csv', transforms_train)

train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size

print("Train size : {} / Test size : {}".format(train_size, test_size))

train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32)
# train_loader = DataLoader(full_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=16)

# Prepare Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MnistModel().to(device)
# print(summary(model, input_size=(1, 3, 256, 256), verbose=0))

# Optimizer, loss
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
lambda1 = lambda epoch: 0.94 ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
criterion = nn.BCELoss()

# Train
num_epochs = 200
model.train()
best_loss = 1e8
for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        images = images.to(device)
        targets = targets.to(device)

        output_prob, output_cnt = model(images)
        loss1 = criterion(output_prob, targets[:, :-1])
        loss2 = 0.1 * nn.SmoothL1Loss()(torch.sum(output_prob, 1).squeeze(), targets[:, -1:])
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()
        
        # Log and save
        if (i+1) % 10 == 0:
            output_prob = output_prob > 0.5
            acc = (output_prob == targets[:, :-1]).float().mean()
            neptune.log_metric('train loss', loss.item())
            neptune.log_metric('train accuracy', acc.item())
            print(f'Epoch {epoch} / Step: {i+1}: Train loss1 {loss1.item():.5f}, loss2 {loss2.item():.5f} Train Accuracy {acc.item():.5f}')

    scheduler.step()
    
    val_loss, val_acc = validate(test_loader, model, criterion, epoch, device)        
    is_best = val_loss < best_loss
    best_loss = min(val_loss, best_loss)

    print(f'Epoch {epoch}: Val loss {val_loss:.5f}, Val Accuracy {val_acc:.5f}')
    neptune.log_metric('validation loss', val_loss)
    neptune.log_metric('validation accuracy', val_acc)
    torch.save(model.state_dict(), 'data/recent.pth')

    # if is_best:
    torch.save(model.state_dict(), f'data/model-{epoch}.pth')


