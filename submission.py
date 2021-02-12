import torch
import pandas as pd

from torchinfo import summary
from torch.utils.data import DataLoader

from model import MnistModel
from dataset import MnistDataset
from model import MnistModel
from preprocess import transforms_train, transforms_test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('best.pth').to(device)
print(summary(model, input_size=(1, 3, 256, 256), verbose=0))

submit = pd.read_csv('data/sample_submission.csv')

model.eval()

testset = MnistDataset(
    'data/test', 'data/sample_submission.csv', transforms_test)
test_loader = DataLoader(testset, batch_size=32, num_workers=4)


batch_size = test_loader.batch_size
batch_index = 0
for i, (images, targets) in enumerate(test_loader):
    images = images.to(device)
    targets = targets.to(device)
    outputs = model(images)
    outputs = outputs > 0.5
    batch_index = i * batch_size
    submit.iloc[batch_index:batch_index+batch_size, 1:] = \
        outputs.long().squeeze(0).detach().cpu().numpy()

submit.to_csv('submit.csv', index=False)