# Evaluation Results - Batch 1 - Model 1

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
- Average Loss: 12.6241
- Translation RMSE: 0.2008
- Translation Accuracy: 0.20 cm
- Rotation RMSE: 120.5671
- Rotation Accuracy: 120.57°

### Test Set
- Average Loss: 12.6582
- Translation RMSE: 0.2037
- Translation Accuracy: 0.20 cm
- Rotation RMSE: 120.8046
- Rotation Accuracy: 120.80°

## Dataset Statistics
### Training Set
- Translation range: [-0.08, 0.33] m

### Validation Set
- Translation range: [-0.04, 0.30] m

### Test Set
- Translation range: [-0.10, 0.33] m

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch1
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/ViT6DP_batch1.2.pth
