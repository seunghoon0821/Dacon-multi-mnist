import torch
import pandas as pd
import numpy as np

from torchinfo import summary
from torch.utils.data import DataLoader

from model import MnistModel
from dataset import MnistDataset
from preprocess import transforms_train, transforms_test
from tta import TTA

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Get Model
model = MnistModel()
model.load_state_dict(torch.load('data/model-25.pth', map_location=device))
# print(summary(model, input_size=(1, 3, 256, 256), verbose=0))
model.to(device)
model.eval()

# Prepare Data
submit = pd.read_csv('data/sample_submission.csv')
testset = MnistDataset(
    'data/test', 'data/sample_submission.csv', transforms_test)
test_loader = DataLoader(testset, batch_size=8, num_workers=4)

# Test time augmentation
conf = '{"augs":["NO",\
                "ROT90",\
                "ROT180",\
                "ROT270"],\
        "mean":"ARITH"}'

class mymodel:
    def __init__(self):
        self.model = model 
    def __call__(self, inputs):
        outputs, _ = self.model(inputs)
        return outputs

models = TTA(mymodel(), device, conf)
    
# Inference
batch_size = test_loader.batch_size
batch_index = 0
for i, (images, targets) in enumerate(test_loader):
    # images.to(device)
    images = images.permute(0, 2, 3, 1)
    images = images.numpy()

    outputs = models.predict_images(images)
    outputs = torch.tensor(outputs)
    ans = np.zeros([outputs.shape[0], 26])
    for j in range(outputs.shape[0]):
        # val, idx = torch.topk(outputs[j], int(torch.sum(outputs[j])))
        for k in range(26):
            if outputs[j][k] > 0.6:
                ans[j][k] = 1
    batch_index = i * batch_size
    submit.iloc[batch_index:batch_index+batch_size, 1:] = \
        ans.astype(np.long)

# Make submission file
submit.to_csv('data/submit.csv', index=False)
