import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import Dataset
from utility.train_utils import make_head # Relative import to train_utils.py

class StereoConvNeXt6DP(nn.Module):
    """Shared weight twin head model for stereo image input."""
    def __init__(self, head_layers=[256], pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "convnextv2_nano.fcmae_ft_in22k_in1k",
            pretrained=pretrained,
            features_only=False
        )
        self.backbone.head = nn.Identity() # Remove classifier
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        # Input dimension for the head will be 2 * backbone output feature size
        self.concat_dim = 640 * 2 # ConvNeXtV2 Nano output features is 640
        self.head = make_head(self.concat_dim, 9, head_layers)

    def extract_features(self, x):
        """Extracts features from a single image using the shared backbone."""
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def forward(self, xL, xR):
        """Processes left and right images, concatenates features, and passes through the head."""
        fL = self.extract_features(xL)
        fR = self.extract_features(xR)
        f = torch.cat([fL, fR], dim=1) # Concatenate features
        return self.head(f)
    
class StereoPoseDataset(Dataset):
    """Dataset for stereo image input (SW-Twin model). Splits a single image into left and right."""
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        mid = width // 2
        imageL = image.crop((0, 0, mid, height))
        imageR = image.crop((mid, 0, width, height))
        label = torch.tensor(self.annotations.iloc[idx, 1:].astype(np.float32).values)
        if self.transform:
            imageL = self.transform(imageL)
            imageR = self.transform(imageR)
        return imageL, imageR, label