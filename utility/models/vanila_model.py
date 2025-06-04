import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import Dataset
from utility.train_utils import make_head # Relative import to train_utils.py


class ConvNeXt6DP(nn.Module):
    """Vanilla model for single image input."""
    def __init__(self, head_layers=[512], pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "convnextv2_nano.fcmae_ft_in22k_in1k",
            pretrained=pretrained,
            features_only=False
        )
        self.backbone.head = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = make_head(640, 9, head_layers)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x)

class PoseDataset(Dataset):
    """Dataset for single image input (Vanilla model)."""
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.annotations.iloc[idx, 1:].astype(np.float32).values)
        if self.transform:
            image = self.transform(image)
        return image, label