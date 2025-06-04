import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import Dataset
from utility.train_utils import make_head # Relative import to train_utils.py


class ConvNeXt6DP6ch(nn.Module):
    """Model for 6-channel stacked image input."""
    def __init__(self, head_layers=[512], pretrained=True, in_chans=6):
        super().__init__()
        self.backbone = timm.create_model(
            "convnextv2_nano.fcmae_ft_in22k_in1k",
            pretrained=pretrained,
            features_only=False,
            in_chans=in_chans # timm handles this for some models, but we'll inflate manually too
        )
        self.backbone.head = nn.Identity()
        self._inflate_input_channels(in_chans)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = make_head(640, 9, head_layers)

    def _inflate_input_channels(self, in_chans):
        # Adapts the first convolutional layer to accept `in_chans` input channels.
        old_conv = self.backbone.stem[0]
        new_conv = nn.Conv2d(
            in_chans,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        with torch.no_grad():
            if in_chans == 6:
                # Replicate weights for 6 channels (assuming 3+3 structure like stereo)
                new_conv.weight[:, :3] = old_conv.weight.clone()
                new_conv.weight[:, 3:] = old_conv.weight.clone()
            else:
                # Fallback for other channel counts, simply repeat the 3-channel weights
                repeat = in_chans // old_conv.in_channels
                for i in range(repeat):
                    new_conv.weight[:, i*old_conv.in_channels:(i+1)*old_conv.in_channels] = old_conv.weight.clone()
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self.backbone.stem[0] = new_conv

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x)
    
class Stereo6ChPoseDataset(Dataset):
    """Dataset for 6-channel stacked image input (6ch model). Stacks left and right images."""
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

        if self.transform:
            imageL = self.transform(imageL)  # (3, H, W)
            imageR = self.transform(imageR)  # (3, H, W)

        # Stack to make (6, H, W)
        image6ch = torch.cat([imageL, imageR], dim=0)
        label = torch.tensor(self.annotations.iloc[idx, 1:].astype(np.float32).values)
        return image6ch, label