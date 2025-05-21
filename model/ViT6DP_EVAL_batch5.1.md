# Evaluation Results - Batch 5 - Model 1

## Training Configuration
- Batch Size: 32
- Epochs: 20
- Learning Rate: 0.0001
- Translation Weight : 1.5
- Rotation Weight : 1.0
- Angular Weight : 0.1
- Patience : 3
- Image Size: 224
- Device: cuda
- Optimizer : Adam

## Model Architecture
- Backbone: ViT Base Patch16 224
- Head: Linear(768->512->9)

## Evaluation Metrics

### Validation Set
- Average Loss: 11.7557
- Translation RMSE: 0.0894
- Translation Accuracy: 0.09 cm
- Rotation RMSE: 113.5206
- Rotation Accuracy: 113.52°

### Test Set
- Average Loss: 11.5332
- Translation RMSE: 0.0894
- Translation Accuracy: 0.09 cm
- Rotation RMSE: 111.3530
- Rotation Accuracy: 111.35°

## Dataset Statistics
### Training Set
- Translation range: [-0.08, 0.53] m

### Validation Set
- Translation range: [-0.06, 0.52] m

### Test Set
- Translation range: [-0.07, 0.52] m

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch5
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/ViT6DP_batch5.1.pth
