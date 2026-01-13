import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import pandas as pd
import os
from torchvision import transforms

# --- KONFIGURASI ---

# Tentukan path ke model-model yang sudah di-freeze (.pt)
MODEL_CONFIGS = {
    "BASELINE": {
        "path": "CLEAN-S-ConvNeXt6DP7.10.pt",
        "type": "VANILA"
    },
    "SNN": {
        "path": "CLEAN-S-SW-ConvNeXt6DP7.8.pt",
        "type": "DOUBLE"
    },
    "EARLY_FUSION": {
        "path": "CLEAN-S6ch-ConvNeXt6DP7.9.pt",
        "type": "6ch"
    }
}

# Tentukan gambar sampel dan ground truth-nya
SAMPLE_IMAGE_PATH = "dataset/batch7/val/images/frame_0002_v0.png" # Contoh gambar
GROUND_TRUTH_LABEL = [-0.05480006078964628,0.09807449401359966,0.5039512274910102,0.9427187740350456,0.17005579309663155,0.19009960291854772,-0.9808108247847706,0.2741230636996277,0.09534753913601364]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# --- FUNGSI PREPROCESSING ---

def get_transform():
    """Mendefinisikan transformasi standar untuk gambar."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])

# --- FUNGSI UTAMA ---

def run_comparison():
    """
    Memuat model, menjalankan inferensi pada gambar sampel,
    dan menampilkan perbandingan hasil prediksi.
    """
    print(f"Menjalankan perbandingan pada perangkat: {DEVICE}")
    print(f"Gambar Sampel: {SAMPLE_IMAGE_PATH}\n")

    try:
        image = Image.open(SAMPLE_IMAGE_PATH).convert("RGB")
    except FileNotFoundError:
        print(f"ERROR: File gambar tidak ditemukan di '{SAMPLE_IMAGE_PATH}'. Pastikan path sudah benar.")
        return

    transform = get_transform()
    results = [['GROUND_TRUTH'] + GROUND_TRUTH_LABEL]

    for model_name, config in MODEL_CONFIGS.items():
        print(f"Memproses model: {model_name}...")
        
        try:
            from utility.inference_model import get_model
            # PERBAIKAN: Muat model terlebih dahulu
            model = get_model(config["path"]) 
            # PERBAIKAN: Kemudian pindahkan ke device
            model.to(DEVICE)
            model.eval()
        except (FileNotFoundError, ImportError) as e:
            print(f"  -> ERROR: Gagal memuat model di '{config['path']}'. Pastikan path dan file utility benar. Error: {e}")
            continue

        with torch.no_grad():
            if config["type"] == "VANILA":
                input_tensor = transform(image).unsqueeze(0).to(DEVICE)
                prediction = model(input_tensor)
            
            elif config["type"] == "DOUBLE":
                width, height = image.size
                mid = width // 2
                imageL = image.crop((0, 0, mid, height))
                imageR = image.crop((mid, 0, width, height))
                
                tensorL = transform(imageL).unsqueeze(0).to(DEVICE)
                tensorR = transform(imageR).unsqueeze(0).to(DEVICE)
                prediction = model(tensorL, tensorR)

            elif config["type"] == "6ch":
                width, height = image.size
                mid = width // 2
                imageL = image.crop((0, 0, mid, height))
                imageR = image.crop((mid, 0, width, height))
                
                tensorL = transform(imageL)
                tensorR = transform(imageR)
                
                input_tensor = torch.cat([tensorL, tensorR], dim=0).unsqueeze(0).to(DEVICE)
                prediction = model(input_tensor)
        
        output_vars = prediction.cpu().numpy().flatten().tolist()
        results.append([model_name] + output_vars)

    columns = ['Source', 'x', 'y', 'z', 'r1x', 'r1y', 'r1z', 'r2x', 'r2y', 'r2z']
    df = pd.DataFrame(results, columns=columns)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print("\n--- HASIL PERBANDINGAN PREDIKSI ---")
    print(df.to_string(index=False))
    print("------------------------------------")
    df.to_csv('comparison.csv', index=False)

if __name__ == "__main__":
    run_comparison()