import os
from typing import Tuple, Sequence, Callable
import numpy as np
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import csv
from PIL import Image


class MnistDataset(Dataset):
    def __init__(self, dir: os.PathLike, image_ids: os.PathLike, transforms: Sequence[Callable], label_smoothing=False, smoothing_val=0.05) -> None:
        self.dir = dir
        self.transforms = transforms
        self.label_smoothing = label_smoothing
        self.smoothing_val = smoothing_val
        self.labels = {}
        with open(image_ids, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.labels[int(row[0])] = list(map(int, row[1:]))

        self.image_ids = list(self.labels.keys())

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_id = self.image_ids[index]
        image = Image.open(
            os.path.join(
                self.dir, f'{str(image_id).zfill(5)}.png')).convert('RGB')

        if self.label_smoothing:
            target = np.array(self.labels.get(image_id).float(
            ) * (1 - self.smoothing_val) + 0.5 * self.smoothing_val).astype(np.float32)
        else:
            target = np.array(self.labels.get(image_id)).astype(np.float32)

        if self.transforms is not None:
            image = np.array(image)
            image = self.transforms(image=image)['image']
            # image = self.transforms(image)
        return image, target
