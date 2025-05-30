import torch
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "baked")

def get_model(name):
    if not name.endswith(".pt"):
        raise ValueError("Only TorchScript .pt models are supported in this mode.")
    path = os.path.join(MODEL_DIR, name)
    model = torch.jit.load(path, map_location=DEVICE)
    model.eval()
    return model
