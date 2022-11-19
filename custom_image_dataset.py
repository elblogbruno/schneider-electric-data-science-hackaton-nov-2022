import os
import pandas as pd
from torchvision.io import read_image
import torch
import cv2

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = annotations_file.dataset
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # print("idx", idx)
        img_path = self.img_labels.iloc[idx, 3]

        # print("img_path", img_path)
        
        # image = read_image(img_path)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (512, 512))
        label = self.img_labels.iloc[idx, 4]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label