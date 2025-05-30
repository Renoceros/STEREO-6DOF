import os
import torch
import torch.nn as nn
import timm

# === MANUAL SELECTION ===
BATCH_ID = 7
MODEL_ID = 5
MODEL_VER = "S"
MODEL_NAME = f"CLEAN-{MODEL_VER}-ConvNeXt6DP{BATCH_ID}.{MODEL_ID}.pth"  # Replace this

# === CONFIG ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "model"
OUTPUT_DIR = os.path.join(MODEL_DIR, "baked")
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_SIZE = 244


# === MODELS ===
class ConvNeXt6DP(nn.Module):
    def __init__(self):
        super(ConvNeXt6DP, self).__init__()
        self.backbone = timm.create_model(
            "convnextv2_nano.fcmae_ft_in22k_in1k",
            pretrained=True,
            features_only=False
        )
        self.backbone.head = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Linear(256, 9)            
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Linear(128, 9)
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x)


class ConvNeXt6DP6ch(nn.Module):
    def __init__(self):
        super(ConvNeXt6DP6ch, self).__init__()
        self.backbone = timm.create_model(
            "convnextv2_nano.fcmae_ft_in22k_in1k",
            pretrained=True,
            features_only=False
        )
        self.backbone.head = nn.Identity()
        self._inflate_input_channels()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(640, 256),
            nn.ReLU(),
            nn.Linear(256, 9)#,
            # nn.ReLU(),
            # nn.Linear(128, 9)
        )

    def _inflate_input_channels(self):
        old_conv = self.backbone.stem[0]
        new_conv = nn.Conv2d(
            in_channels=6,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight.clone()
            new_conv.weight[:, 3:] = old_conv.weight.clone()
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)
        self.backbone.stem[0] = new_conv

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.head(x)


class StereoConvNeXt6DP(nn.Module):
    def __init__(self):
        super(StereoConvNeXt6DP, self).__init__()
        self.backbone = timm.create_model(
            "convnextv2_nano.fcmae_ft_in22k_in1k",
            pretrained=True,
            features_only=False
        )
        self.backbone.head = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(640 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 9)
        )

    def extract_features(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def forward(self, xL, xR):
        fL = self.extract_features(xL)
        fR = self.extract_features(xR)
        return self.head(torch.cat([fL, fR], dim=1))


# === HELPERS ===
def get_model_from_name(name):
    name = name.lower()
    if "6ch" in name:
        return ConvNeXt6DP6ch()
    elif "sw" in name:
        return StereoConvNeXt6DP()
    else:
        return ConvNeXt6DP()


def get_dummy_input(name):
    name = name.lower()
    if "6ch" in name:
        return torch.randn(1, 6, IMG_SIZE, IMG_SIZE).to(DEVICE)
    elif "sw" in name:
        left = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        right = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
        return (left, right)
    else:
        return torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)


# === MAIN ===
def main():
    fpath = os.path.join(MODEL_DIR, MODEL_NAME)
    print(f"Baking model: {MODEL_NAME}")

    model = get_model_from_name(MODEL_NAME)
    model.load_state_dict(torch.load(fpath, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    dummy_input = get_dummy_input(MODEL_NAME)

    try:
        scripted = torch.jit.trace(model, dummy_input)
    except Exception as e:
        print(f"❌ Failed to trace {MODEL_NAME}: {e}")
        return

    outname = MODEL_NAME.replace(".pth", ".pt")
    outpath = os.path.join(OUTPUT_DIR, outname)
    scripted.save(outpath)
    print(f"✅ TorchScript model saved to: {outpath}")


if __name__ == "__main__":
    main()
