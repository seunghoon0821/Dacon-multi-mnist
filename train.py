import torch
from torch.utils.data import DataLoader
from torch import nn
from dataset import MnistDataset
from preprocess import transforms_train, transforms_test
from model import MnistModel
from torchinfo import summary
import torch.optim as optim
import neptune

# Init Neptune
neptune.init(project_qualified_name='dongkyuk/dacon-mnist',
             api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMTlmOGExYWUtNDRlOS00MTk1LThiOTQtOGY4MDkyZDAxZjY2In0=',
             )

neptune.create_experiment()

# Prepare Data
trainset = MnistDataset(
    'data/train', 'data/dirty_mnist_2nd_answer.csv', transforms_train)
train_loader = DataLoader(trainset, batch_size=32, num_workers=8)

# Prepare Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MnistModel().to(device)
print(summary(model, input_size=(1, 3, 256, 256), verbose=0))

# Optimizer, loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MultiLabelSoftMarginLoss()

# Train
num_epochs = 10
model.train()

for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        images = images.to(device)
        targets = targets.to(device)

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # Log and save
        if (i+1) % 10 == 0:
            outputs = outputs > 0.5
            acc = (outputs == targets).float().mean()
            print(f'Epoch {epoch}: Train loss {loss.item():.5f}, Train Accuracy {acc.item():.5f}')
            neptune.log_metric('train loss', loss.item())
            neptune.log_metric('train accuracy', acc.item())
            torch.save(model, 'best.pth')


