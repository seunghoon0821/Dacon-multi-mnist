import torch
import torch.optim as optim
import neptune

from torch.utils.data import DataLoader
from torch import nn
from dataset import MnistDataset
from preprocess import transforms_train, transforms_test
from model import MnistModel
from torchinfo import summary
from trainval import train, validate
from sklearn.model_selection import train_test_split

#Init Neptune
neptune.init(project_qualified_name='dongkyuk/dacon-mnist',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMTlmOGExYWUtNDRlOS00MTk1LThiOTQtOGY4MDkyZDAxZjY2In0=',
             )

# neptune.init(project_qualified_name='dhdroid/Dacon-MNIST',
#              api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZWM3ZDFmYjAtM2FlNS00YzUzLThjYTgtZjU3ZmM1MzJhOWQ4In0=',
#              )
neptune.create_experiment()

# cuda cache 초기화
torch.cuda.empty_cache()

# Prepare Data
full_dataset = MnistDataset(
    'data/train', 'data/dirty_mnist_2nd_answer.csv', transforms_train)

train_size = int(0.9 * len(full_dataset))
test_size = len(full_dataset) - train_size

print("Train size : {} / Test size : {}".format(train_size, test_size))

train_dataset, test_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=16)

# Prepare Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MnistModel().to(device)
# model.load_state_dict(torch.load('data/best.pth', map_location=device))
print(summary(model, input_size=(1, 3, 256, 256), verbose=0))

# Optimizer, loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MultiLabelSoftMarginLoss()

# Train
num_epochs = 100
best_loss = 1
for epoch in range(num_epochs):
    # Train
    train(train_loader, model, optimizer, criterion, epoch, device)

    # Validate
    val_loss, val_acc = validate(test_loader, model, criterion, epoch, device)

    # Save recent
    torch.save(model.state_dict(), 'data/recent.pth')

    # Save best
    is_best = val_loss < best_loss
    best_loss = min(val_loss, best_loss)
    if is_best:
        torch.save(model.state_dict(), 'data/best.pth')
