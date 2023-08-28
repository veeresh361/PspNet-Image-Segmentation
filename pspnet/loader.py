import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from config import IMG_SIZE


class SegmentationDataset(Dataset):
    def __init__(self, df, augmentations):
        self.df = df
        self.augmentations = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row.images
        mask_path = row.masks

        image = cv2.imread(image_path)  # bgr
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # bgr -> rgb
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
        mask = np.expand_dims(mask, axis=-1)

        if self.augmentations:
            data = self.augmentations(image=image, mask=mask)
            image = data["image"]
            mask = data["mask"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

        image = torch.Tensor(image) / 255.0
        mask = torch.round(torch.Tensor(mask) / 255.0)

        return image, mask


class TestSegmentationDataset(Dataset):
    def __init__(self, df, augmentations):
        self.df = df
        self.augmentations = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row.images

        image = cv2.imread(image_path)  # bgr
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # bgr -> rgb
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

        if self.augmentations:
            data = self.augmentations(image=image)
            image = data["image"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        image = torch.Tensor(image) / 255.0

        return image
