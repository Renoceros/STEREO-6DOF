# 📦 1. Imports
import os
import json
import csv
import time
import torch
import timm
import torch.nn as nn
from torch.nn.utils import prune
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from typing import Tuple
from torch.amp import GradScaler, autocast

# 🧾 2. Declarations / Configuration
with open("GlobVar.json", "r") as file:
    gv = json.load(file)

mod_id = gv['mod_id']

BATCH_ID = 5
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
TRANS_WEIGHT = 1.5
ROTATION_WEIGHT = 1.0
ANGULAR_WEIGHT = 0.1
PATIENCE = 3
IMG_SIZE = 224
BASE_DIR = os.path.expanduser("~/SKRIPSI/SCRIPTS")
DATASET_DIR = os.path.join(BASE_DIR, f"dataset/batch{BATCH_ID}")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, f"model/S-ConvNeXt6DP{BATCH_ID}.{mod_id}.pth")
BEST_MODEL_PATH = os.path.join(BASE_DIR, f"model/BS-ConvNeXt6DP{BATCH_ID}.{mod_id}.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📁 3. Dataset Class
class PoseDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(self.annotations.iloc[idx, 1:].values, dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, label

# 🧮 4. Rotation Conversion + Loss Functions
def rotation_error(R_pred, R_gt):
    """Compute angular error in degrees between rotation matrices."""
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = torch.diagonal(R_diff, dim1=1, dim2=2).sum(dim=1)
    eps = 1e-6
    angle_rad = torch.acos(torch.clamp((trace - 1) / 2, min=-1 + eps, max=1 - eps))
    return torch.rad2deg(angle_rad)


def geodesic_loss(R_pred, R_gt):
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = torch.diagonal(R_diff, dim1=1, dim2=2).sum(dim=1)
    eps = 1e-6
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1 + eps, 1 - eps))
    return angle.mean()

def compute_rotation_matrix_from_ortho6d(poses_6d):
    """Convert 6D rotation representation to 3x3 rotation matrices."""
    x_raw = poses_6d[:, 0:3]
    y_raw = poses_6d[:, 3:6]

    x = F.normalize(x_raw, dim=1)
    z = F.normalize(torch.cross(x, y_raw, dim=1), dim=1)
    y = torch.cross(z, x, dim=1)

    rot = torch.stack((x, y, z), dim=-1)  # Shape: [B, 3, 3]
    return rot

def combined_loss(output, target, trans_w=1.0, rot_w=1.0, ang_w=0.1):
    pred_trans = output[:, :3]
    gt_trans = target[:, :3]
    pred_rot_6d = output[:, 3:9]
    gt_rot_6d = target[:, 3:9]

    pred_rot = compute_rotation_matrix_from_ortho6d(pred_rot_6d)
    gt_rot = compute_rotation_matrix_from_ortho6d(gt_rot_6d)

    loss_trans = F.mse_loss(pred_trans, gt_trans)
    loss_rot = geodesic_loss(pred_rot, gt_rot)
    return trans_w * loss_trans + rot_w * loss_rot


def rotation_error_deg_from_6d(pred_6d, gt_6d):
    R_pred = compute_rotation_matrix_from_ortho6d(pred_6d)
    R_gt = compute_rotation_matrix_from_ortho6d(gt_6d)

    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(cos_theta)
    return torch.rad2deg(theta).mean()

def compute_errors(outputs, labels):
    pred_trans = outputs[:, :3]
    gt_trans = labels[:, :3]

    pred_rot_6d = outputs[:, 3:9]
    gt_rot_6d = labels[:, 3:9]

    trans_rmse = torch.sqrt(F.mse_loss(pred_trans, gt_trans))
    rot_rmse = rotation_error_deg_from_6d(pred_rot_6d, gt_rot_6d)
    return trans_rmse.item(), rot_rmse.item()

def calculate_translation_rmse(preds, gts):
    trans_rmse = np.sqrt(np.mean(np.sum((preds[:, :3] - gts[:, :3])**2, axis=1)))
    return trans_rmse * 100  # Convert m → cm

def translation_accuracy_percentage(rmse_cm, range_cm):
    return max(0.0, 100.0 * (1 - rmse_cm / range_cm))


# 🖼 5. Transform & Dataloaders
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

def get_dataloader(split):
    csv_path = os.path.join(DATASET_DIR, f"{split}.csv")
    images_dir = os.path.join(DATASET_DIR, split)
    dataset = PoseDataset(csv_path, images_dir, transform=get_transform())
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(split == "train"))

# 📊 6. Dataset Stats
def get_dataset_stats(loader):
    translations = []
    for _, labels in loader:
        translations.append(labels[:, :3])
    trans = torch.cat(translations, dim=0)
    return {
        'min': trans.min(dim=0).values,
        'max': trans.max(dim=0).values,
        'mean': trans.mean(dim=0),
        'std': trans.std(dim=0)
    }

train_loader = get_dataloader("train")
val_loader = get_dataloader("val")
test_loader = get_dataloader("test")


# 🧠 8. Model Definition
class ConvNeXt6DP(nn.Module):
    def __init__(self):
        super(ConvNeXt6DP, self).__init__()
        self.backbone = timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k", pretrained=True)
        self.backbone.head = nn.Identity()  # Remove the classification head
        self.head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 9)  # 3 translation + 6 rotation
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)

# 🔧 9. Model Setup
model = ConvNeXt6DP().to(DEVICE)  # Use the updated model class
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

# 🏋️ 10. Training Function
def train(validate=True, resume_from_checkpoint=False):
    scaler = GradScaler()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    now = [time.time()]
    start_epoch = 0

    # Resume from checkpoint if specified
    if resume_from_checkpoint:
        checkpoint = torch.load(MODEL_SAVE_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        start_epoch = checkpoint.get('epoch', 0)
        print(f"✅ Resumed from checkpoint at epoch {start_epoch}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        print("\n")
        print(f"📦 EPOCH : {epoch + 1}")
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=False)

        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                outputs = model(images)
                loss = combined_loss(outputs, labels, TRANS_WEIGHT, ROTATION_WEIGHT, ANGULAR_WEIGHT)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        print(f"✅ Epoch {epoch + 1} Avg Training Loss: {avg_train_loss:.4f}")
        now.append(time.time())
        print(f"⏱️ Time per epoch {epoch + 1}: {int(now[epoch + 1] - now[epoch])}s")

        # Apply structured pruning periodically (skip epoch 0)
        if epoch > 0 and epoch % 5 == 0:
            parameters_to_prune = (
                (module, 'weight') for module in model.modules()
                if isinstance(module, (nn.Linear, nn.Conv2d))
            )
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=0.1
            )
            print(f"⚠️ Pruning applied at epoch {epoch}")

        if validate:
            model.eval()
            val_loss = 0.0
            total_trans_rmse, total_rot_rmse = 0.0, 0.0

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    outputs = model(images)

                    loss = combined_loss(outputs, labels, TRANS_WEIGHT, ROTATION_WEIGHT, ANGULAR_WEIGHT)
                    val_loss += loss.item()

                    trans_rmse, rot_rmse = compute_errors(outputs, labels)
                    total_trans_rmse += trans_rmse
                    total_rot_rmse += rot_rmse

            avg_val_loss = val_loss / len(val_loader)
            avg_trans_rmse = total_trans_rmse / len(val_loader)
            avg_rot_rmse = total_rot_rmse / len(val_loader)

            print(f"📉 Validation Loss: {avg_val_loss:.4f}")
            print(f"📐 RMSE - Translation: {avg_trans_rmse:.4f}, Rotation: {avg_rot_rmse:.4f}")

            scheduler.step(avg_val_loss)

            # Save best model if validation improves
            if epoch != 0 and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'epochs_no_improve': epochs_no_improve,
                    'epoch': epoch + 1
                }, BEST_MODEL_PATH)
                print("💾 Best model saved.")
            else:
                epochs_no_improve += 1
                print(f"📉 No improvement ({epochs_no_improve}/{PATIENCE})")

                if epochs_no_improve >= PATIENCE:
                    print("⏹️ Early stopping triggered")
                    break

        # Always save last checkpoint
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'epochs_no_improve': epochs_no_improve,
            'epoch': epoch + 1
        }, MODEL_SAVE_PATH)




# ▶️ 11. Run Training
train(validate=True,resume_from_checkpoint=False)

# 💾 12. Save Model + Update Config

gv['mod_id'] += 1
with open("GlobVar.json", "w") as file:
    json.dump(gv, file, indent=4)
print("mod_id updated in GlobVar.json")

# 🧪 13. Test Function
def test_model(model, loader, mode='Test', use_amp=False):
    inference_times = []
    model.eval()
    total_loss, total_trans_rmse, total_rot_rmse = 0, 0, 0
    all_preds, all_gts = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Running {mode}"):
            
            start_time = time.time()
            outputs = model(images)
            inference_time = (time.time() - start_time) * 1000 / images.size(0)  # ms per image
            inference_times.append(inference_time)
            
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Optionally use mixed precision for testing
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = combined_loss(outputs, labels, TRANS_WEIGHT, ROTATION_WEIGHT, ANGULAR_WEIGHT)
            else:
                outputs = model(images)
                loss = combined_loss(outputs, labels, TRANS_WEIGHT, ROTATION_WEIGHT, ANGULAR_WEIGHT)

            total_loss += loss.item()
            trans_rmse, rot_rmse = compute_errors(outputs, labels)
            total_rot_rmse += rot_rmse
            total_trans_rmse += trans_rmse

            all_preds.append(outputs.cpu().numpy())
            all_gts.append(labels.cpu().numpy())
    avg_inference_time = np.mean(inference_times)
    avg_loss = total_loss / len(loader)
    return avg_loss, total_trans_rmse, total_rot_rmse, np.concatenate(all_preds), np.concatenate(all_gts), avg_inference_time


# 🔍 14. Run Test
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.to(DEVICE)
test_avg_loss, test_total_trans_rmse, test_total_rot_rmse, test_preds, test_gts, test_avg_inference_time = test_model(model, test_loader, mode='Test', use_amp=True)


# 🧪 15. Validation Function
def validate_model(model_path=None):
    inference_times = []
    if model_path:
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} does not exist.")
        model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()
    total_loss, total_trans_rmse, total_rot_rmse = 0, 0, 0
    all_preds, all_gts = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Running Validation"):
            
            start_time = time.time()
            outputs = model(images)
            inference_time = (time.time() - start_time) * 1000 / images.size(0)  # ms per image
            inference_times.append(inference_time)

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.cuda.amp.autocast():  # Optional for mixed precision inference
                outputs = model(images)
                loss = combined_loss(outputs, labels, TRANS_WEIGHT, ROTATION_WEIGHT, ANGULAR_WEIGHT)

            total_loss += loss.item()
            trans_rmse, rot_rmse = compute_errors(outputs, labels)
            total_rot_rmse += rot_rmse
            total_trans_rmse += trans_rmse

            all_preds.append(outputs.cpu().numpy())
            all_gts.append(labels.cpu().numpy())

    avg_inference_time = np.mean(inference_times)
    avg_loss = total_loss / len(val_loader)
    return avg_loss, total_trans_rmse, total_rot_rmse, np.concatenate(all_preds), np.concatenate(all_gts), avg_inference_time


# 🔄 16. Run Validation
val_avg_loss, val_total_trans_rmse, val_total_rot_rmse, val_preds, val_gts, val_avg_inference_time = validate_model(MODEL_SAVE_PATH)


# 🧹 17. Clear CUDA Cache
torch.cuda.empty_cache()

# 📐 18. Compute Accuracy

test_translation_rmse_cm = calculate_translation_rmse(test_preds, test_gts)
val_translation_rmse_cm = calculate_translation_rmse(val_preds, val_gts)
test_rot_accuracy = rotation_error_deg_from_6d(torch.tensor(test_preds[:, 3:]), torch.tensor(test_gts[:, 3:]))
val_rot_accuracy = rotation_error_deg_from_6d(torch.tensor(val_preds[:, 3:]), torch.tensor(val_gts[:, 3:]))

train_trans_stats = get_dataset_stats(train_loader)
val_trans_stats = get_dataset_stats(val_loader)
test_trans_stats = get_dataset_stats(test_loader)

test_range_cm = (test_trans_stats['max'].mean() - test_trans_stats['min'].mean()) * 100
val_range_cm = (val_trans_stats['max'].mean() - val_trans_stats['min'].mean()) * 100

test_trans_accuracy_pct = translation_accuracy_percentage(test_translation_rmse_cm, test_range_cm)
val_trans_accuracy_pct = translation_accuracy_percentage(val_translation_rmse_cm, val_range_cm)


# 📑 19. Compose Evaluation Report (Markdown)
eval_path = os.path.join(BASE_DIR, f"model/ViT6DP_EVAL_batch{BATCH_ID}.{mod_id-1}.md")
eval_content = f"""# Evaluation Results - Batch {BATCH_ID} - Model {mod_id-1}

## Training Configuration
- Batch Size: {BATCH_SIZE}
- Epochs: {NUM_EPOCHS}
- Learning Rate: {LEARNING_RATE}
- Translation Weight : {TRANS_WEIGHT}
- Rotation Weight : {ROTATION_WEIGHT}
- Angular Weight : {ANGULAR_WEIGHT}
- Patience : {PATIENCE}
- Image Size: {IMG_SIZE}
- Device: {DEVICE}
- Optimizer : Adam

## Model Architecture
- Backbone: Using ConvNeXt
- Head: Linear(768->512->9)

## Evaluation Metrics

### Test Set
- Average Loss: {test_avg_loss:.4f}
- Translation RMSE: {test_total_trans_rmse / len(test_loader):.4f}
- Translation Accuracy: {test_translation_rmse_cm:.2f} cm
- Translation Accuracy %: {test_trans_accuracy_pct:.2f}%
- Rotation RMSE: {test_total_rot_rmse / len(test_loader):.4f}
- Rotation Accuracy: {test_rot_accuracy:.2f}°
- Inference Speed: {test_avg_inference_time:.2f} ms/frame

### Validation Set
- Average Loss: {val_avg_loss:.4f}
- Translation RMSE: {val_total_trans_rmse / len(val_loader):.4f}
- Translation Accuracy: {val_translation_rmse_cm:.2f} cm
- Translation Accuracy %: {val_trans_accuracy_pct:.2f}%
- Rotation RMSE: {val_total_rot_rmse / len(val_loader):.4f}
- Rotation Accuracy: {val_rot_accuracy:.2f}°
- Inference Speed: {val_avg_inference_time:.2f} ms/frame

## Dataset Statistics
### Training Set
- Translation range: [{train_trans_stats['min'].mean():.2f}, {train_trans_stats['max'].mean():.2f}] m

### Validation Set
- Translation range: [{val_trans_stats['min'].mean():.2f}, {val_trans_stats['max'].mean():.2f}] m

### Test Set
- Translation range: [{test_trans_stats['min'].mean():.2f}, {test_trans_stats['max'].mean():.2f}] m

## File Locations
- Dataset Directory: {DATASET_DIR}
- Model Save Path: {MODEL_SAVE_PATH}
"""

with open(eval_path, 'w') as f:
    f.write(eval_content)
print(f"Evaluation report saved to: {eval_path}")

# 📈 20. Write Evaluation to CSV
csv_path = os.path.join(BASE_DIR, "model/eval_results.csv")
write_header = not os.path.exists(csv_path)

csv_data = {
    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_id': BATCH_ID,
    'model_id': mod_id-1,
    'batch_size': BATCH_SIZE,
    'epochs': NUM_EPOCHS,
    'learning_rate': LEARNING_RATE,
    'translation_weight':TRANS_WEIGHT,
    'rotation_weight' : ROTATION_WEIGHT,
    'angular_weight' : ANGULAR_WEIGHT,
    'patience' : PATIENCE,
    'test_loss': test_avg_loss,
    'test_translation_rmse': test_total_trans_rmse / len(test_loader),
    'test_translation_accuracy_pct': test_trans_accuracy_pct,
    'test_rotation_rmse': test_total_rot_rmse / len(test_loader),
    'test_inference_time_ms': test_avg_inference_time,
    'validation_loss': val_avg_loss,
    'validation_translation_rmse': val_total_trans_rmse / len(val_loader),
    'validation_translation_accuracy_pct': val_trans_accuracy_pct,
    'validation_rotation_rmse': val_total_rot_rmse / len(val_loader),
    'validation_inference_time_ms': val_avg_inference_time,
    'model_path': MODEL_SAVE_PATH,
    'eval_path' : eval_path
}

with open(csv_path, 'a', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_data.keys())
    if write_header:
        writer.writeheader()
    writer.writerow(csv_data)
print(f"Results appended to CSV: {csv_path}")