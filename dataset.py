import os
from typing import Tuple, Sequence, Callable
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import csv
from PIL import Image

def label_smooth(p):
    if p > 0.5:
        return p-0.001
    else:
        return 0.001

class MnistDataset(Dataset):
    def __init__(self, dir: os.PathLike, image_ids: os.PathLike, transforms: Sequence[Callable]) -> None:
        self.dir = dir
        self.transforms = transforms

        self.labels = {}
        with open(image_ids, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                label_list = list(map(lambda x: float(x), row[1:]))
                label_list.append(np.sum(label_list))
                self.labels[int(row[0])] = label_list

        self.image_ids = list(self.labels.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_id = self.image_ids[index]
        image = Image.open(
            os.path.join(
                self.dir, f'{str(image_id).zfill(5)}.png')).convert('RGB')
        target = np.array(self.labels.get(image_id)).astype(np.float32)

        if self.transforms is not None:
            image = np.array(image)
            image = self.transforms(image=image)['image']
        return image, target




