# %%
# VIT6D.ipynb refactored for Minty (Linux Mint)
from torch.amp import autocast, GradScaler
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import timm
import time
import datetime
import json
import numpy as np
import csv

# %%
# Load global variables
with open("GlobVar.json", "r") as file:
    gv = json.load(file)

mod_id = gv['mod_id']

# Constants and environment setup
BATCH_ID = 3
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
IMG_SIZE = 224  # Required input size for ViT
BASE_DIR = os.path.expanduser("~/SKRIPSI/SCRIPTS")
DATASET_DIR = os.path.join(BASE_DIR, f"dataset/batch{BATCH_ID}")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, f"model/ViT6DP_batch{BATCH_ID}.{mod_id}.pth")

# Use CUDA if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# %%
class PoseDataset(Dataset):
    def __init__(self, image_dir, label_csv, transform=None):
        self.image_dir = image_dir
        self.labels = pd.read_csv(label_csv)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        img_path = os.path.join(self.image_dir, row['image_name'])
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(row[1:].values.astype('float32'))
        if self.transform:
            image = self.transform(image)
        return image, label

# %%
def sixd_to_rotation_matrix(sixd):
    """Convert 6D rotation representation to 3x3 rotation matrix."""
    a1, a2 = sixd[:, :3], sixd[:, 3:]
    b1 = a1 / torch.norm(a1, dim=1, keepdim=True)
    b2 = a2 - torch.sum(b1 * a2, dim=1, keepdim=True) * b1
    b2 = b2 / torch.norm(b2, dim=1, keepdim=True)
    b3 = torch.cross(b1, b2)
    return torch.stack([b1, b2, b3], dim=-1)

def rotation_error(R_pred, R_gt):
    """Compute angular error in degrees between rotation matrices."""
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = torch.diagonal(R_diff, dim1=1, dim2=2).sum(dim=1)
    angle_rad = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
    return torch.rad2deg(angle_rad)

def combined_loss(pred, target, trans_weight=1.0, rot_weight=1.0, angular_weight=0.1):
    # Split predictions and targets
    pred_trans, pred_rot = pred[:, :3], pred[:, 3:]
    target_trans, target_rot = target[:, :3], target[:, 3:]
    
    # Translation loss (MSE)
    trans_loss = nn.MSELoss()(pred_trans, target_trans)
    
    # Rotation losses
    rot_mse_loss = nn.MSELoss()(pred_rot, target_rot)  # Standard MSE on 6D
    
    # Additional angular error loss
    R_pred = sixd_to_rotation_matrix(pred_rot)
    R_gt = sixd_to_rotation_matrix(target_rot)
    angular_loss = rotation_error(R_pred, R_gt).mean()  # Mean angular error in degrees
    
    # Weighted combination
    total_loss = (trans_weight * trans_loss + 
                 rot_weight * rot_mse_loss + 
                 angular_weight * angular_loss)
    
    return total_loss
# %%
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

def get_dataloader(split):
    image_dir = os.path.join(DATASET_DIR, split, 'images')
    label_csv = os.path.join(DATASET_DIR, split, 'labels.csv')
    dataset = PoseDataset(image_dir, label_csv, transform)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(split == 'train'))

train_loader = get_dataloader('train')
val_loader = get_dataloader('val')
test_loader = get_dataloader('test')

# %%
def compute_errors(pred, target):
    # Translation error (RMSE in meters)
    trans_rmse = torch.sqrt(nn.MSELoss()(pred[:, :3], target[:, :3]))

    # Rotation error (in degrees)
    R_pred = sixd_to_rotation_matrix(pred[:, 3:])
    R_gt = sixd_to_rotation_matrix(target[:, 3:])
    rot_error = rotation_error(R_pred, R_gt).mean()

    return trans_rmse.item(), rot_error.item()

def get_dataset_stats(loader):
    all_labels = []
    for _, labels in loader:
        all_labels.append(labels)
    all_labels = torch.cat(all_labels, dim=0)

    # Translation stats (x,y,z)
    trans_stats = {
        'min': all_labels[:, :3].min(dim=0)[0],
        'max': all_labels[:, :3].max(dim=0)[0],
        'mean': all_labels[:, :3].mean(dim=0),
        'std': all_labels[:, :3].std(dim=0)
    }

    # Rotation stats (Angular)
    rot_stats = {
        'min': all_labels[:, 3:].min(dim=0)[0],
        'max': all_labels[:, 3:].max(dim=0)[0],
        'mean': all_labels[:, 3:].mean(dim=0),
        'std': all_labels[:, 3:].std(dim=0),
        'euler_min': None,  # Placeholder for Euler conversion min
        'euler_max': None   # Placeholder for Euler conversion max
    }

    return trans_stats, rot_stats

# Get stats for each dataset split
train_trans_stats, train_rot_stats = get_dataset_stats(train_loader)
val_trans_stats, val_rot_stats = get_dataset_stats(val_loader)
test_trans_stats, test_rot_stats = get_dataset_stats(test_loader)

print(f"Train Translation stat: {train_trans_stats}      |       Train Rotation stat: {train_rot_stats}")
print(f"Validation Translation stat: {val_trans_stats}     |       Validation Rotation stat: {val_rot_stats}")
print(f"Test Translation stat: {test_trans_stats}      |       Test Rotation stat: {test_rot_stats}")
scaler = GradScaler()
# %%
class ViT6DP(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.backbone.head = nn.Sequential(
            nn.Linear(self.backbone.head.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 9)
        )

    def forward(self, x):
        return self.backbone(x)

# %%
model = ViT6DP().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

# Early stopping
best_val_loss = float('inf')
patience = 3
epochs_no_improve = 0

# %%
def train(validate=True):
    now = [time.time()]
    for epoch in range(NUM_EPOCHS):
        print("\n")
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # AMP forward pass
            with autocast():
                outputs = model(images)
                loss = combined_loss(outputs, labels)
            
            # AMP backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        now.append(time.time())
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}")
        print(f"Time per epoch {epoch + 1}: {int(now[epoch + 1] - now[epoch])}s")

        if validate:
            model.eval()
            val_loss = 0.0
            total_trans_rmse, total_rot_rmse = 0.0, 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(DEVICE)
                    labels = labels.to(DEVICE)
                    outputs = model(images)

                    loss = combined_loss(outputs, labels)
                    val_loss += loss.item()

                    trans_rmse, rot_rmse = compute_errors(outputs, labels)
                    total_trans_rmse += trans_rmse
                    total_rot_rmse += rot_rmse

                avg_val_loss = val_loss / len(val_loader)
                print(f"Val Loss: {avg_val_loss:.4f}")
                print(f"RMSE - Translation: {total_trans_rmse / len(val_loader):.4f}, "
                      f"Rotation: {total_rot_rmse / len(val_loader):.4f}")

                # Scheduler step
                scheduler.step(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == patience:
                        print("Early stopping!")
                        break
        else:
            print("Skipping validation for this epoch.")

# %%
train(validate=True)

# %%
torch.save(model.state_dict(), MODEL_SAVE_PATH)

# Increment version
mod_id += 1
gv['mod_id'] = mod_id
with open("GlobVar.json", "w") as file:
    json.dump(gv, file, indent=4)

# %%
# Function to test the model
def test_model(model, loader, mode='val'):
    model.eval()
    total_loss = 0.0
    total_trans_rmse, total_rot_rmse = 0.0, 0.0
    preds, gts = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader):
            # Optimized device transfer
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            
            # FP32 inference for stable metrics
            outputs = model(images)
            
            loss = combined_loss(outputs, labels)
            trans_rmse, rot_rmse = compute_errors(outputs, labels)
            
            # Accumulate
            total_loss += loss.item()
            total_trans_rmse += trans_rmse
            total_rot_rmse += rot_rmse
            preds.extend(outputs.cpu().numpy())
            gts.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    print(f"{mode.capitalize()} Loss: {avg_loss:.4f}")
    print(f"RMSE - Translation: {total_trans_rmse/len(loader):.4f}, Rotation: {total_rot_rmse/len(loader):.4f}")
    
    return preds, gts, avg_loss, total_trans_rmse, total_rot_rmse# %%
# Load model first
test_model = ViT6DP().to(DEVICE)
test_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

# Then run evaluation
predictions, ground_truths, test_avg_loss, test_total_trans_rmse, test_total_rot_rmse = test_model(
    model=test_model,
    loader=test_loader,
    mode='test'
)
# %%
# Function to validate the model
def validate_model(model_path):
    model = ViT6DP().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    val_total_loss = 0.0
    val_total_trans_rmse, val_total_rot_rmse = 0.0, 0.0
    preds, gts = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)

            loss = combined_loss(outputs, labels)
            val_total_loss += loss.item()

            trans_rmse, rot_rmse = compute_errors(outputs, labels)
            val_total_trans_rmse += trans_rmse
            val_total_rot_rmse += rot_rmse

            preds.extend(outputs.cpu().numpy())
            gts.extend(labels.cpu().numpy())

    val_avg_loss = val_total_loss / len(val_loader)
    print(f"Validation Loss: {val_avg_loss:.4f}")
    print(f"Validation RMSE - Translation: {val_total_trans_rmse / len(val_loader):.4f}, "
          f"Rotation: {val_total_rot_rmse / len(val_loader):.4f}")

    return preds, gts, val_avg_loss, val_total_trans_rmse, val_total_rot_rmse

# %%
val_predictions, val_ground_truths, val_avg_loss, val_total_trans_rmse, val_total_rot_rmse = validate_model(MODEL_SAVE_PATH)

# %%
torch.cuda.empty_cache()

# %%
# Function to calculate translation and rotation accuracy
def calculate_translation_rmse(preds, gts):
    """Euclidean distance between predicted and GT translations (in meters)."""
    errors = np.linalg.norm(preds - gts, axis=1)  # Shape: [N]
    rmse = np.sqrt(np.mean(errors**2))
    return rmse * 1000  # Convert to mm

# %%
val_trans_accuracy, val_rot_accuracy = val_total_trans_rmse / len(val_loader), val_total_rot_rmse / len(val_loader)
test_trans_accuracy, test_rot_accuracy = test_total_trans_rmse / len(test_loader), test_total_rot_rmse / len(test_loader)

# Write evaluation report to markdown
# %%
eval_content = f"""# Evaluation Results - Batch {BATCH_ID}

## Training Configuration
- Batch Size: {BATCH_SIZE}
- Epochs: {NUM_EPOCHS}
- Learning Rate: {LEARNING_RATE}
- Image Size: {IMG_SIZE}
- Device: {DEVICE}
- Optimizer : Adam

## Model Architecture
- Backbone: ViT Base Patch16 224
- Head: Linear(768->512->9)

## Evaluation Metrics

### Validation Set
- Average Loss: {val_avg_loss:.4f}
- Translation RMSE: {val_total_trans_rmse / len(val_loader):.4f}
- Translation Accuracy: {val_trans_accuracy:.2f} cm
- Rotation RMSE: {val_total_rot_rmse / len(val_loader):.4f}
- Rotation Accuracy: {val_rot_accuracy:.2f}°

### Test Set
- Average Loss: {test_avg_loss:.4f}
- Translation RMSE: {test_total_trans_rmse / len(test_loader):.4f}
- Translation Accuracy: {test_trans_accuracy:.2f} cm
- Rotation RMSE: {test_total_rot_rmse / len(test_loader):.4f}
- Rotation Accuracy: {test_rot_accuracy:.2f}°

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

eval_path = os.path.join(BASE_DIR, f"model/ViT6DP_EVAL_batch{BATCH_ID}.{mod_id-1}.md")
with open(eval_path, 'w') as f:
    f.write(eval_content)

print(f"Evaluation report saved to: {eval_path}")

# %%
# Write results to CSV
csv_path = os.path.join(BASE_DIR, "model/eval_results.csv")
write_header = not os.path.exists(csv_path)

csv_data = {
    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'dataset_id': BATCH_ID,
    'model_id': mod_id-1,
    'batch_size': BATCH_SIZE,
    'epochs': NUM_EPOCHS,
    'learning_rate': LEARNING_RATE,
    'test_loss': test_avg_loss,
    'test_translation_rmse': test_total_trans_rmse / len(test_loader),
    'test_rotation_rmse': test_total_rot_rmse / len(test_loader),
    'validation_loss': val_avg_loss,
    'validation_translation_rmse': val_total_trans_rmse / len(val_loader),
    'validation_rotation_rmse': val_total_rot_rmse / len(val_loader),
    'model_path': MODEL_SAVE_PATH,
    'eval_path' : eval_path
}

# Write to CSV
with open(csv_path, 'a', newline='') as csvfile:
    fieldnames = csv_data.keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    if write_header:
        writer.writeheader()
    writer.writerow(csv_data)

print(f"Results appended to CSV: {csv_path}")