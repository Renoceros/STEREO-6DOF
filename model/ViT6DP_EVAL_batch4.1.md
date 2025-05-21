# Evaluation Results - Batch 4 - Model 1

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
- Average Loss: 10.3364
- Translation RMSE: 0.1058
- Translation Accuracy: 0.11 cm
- Rotation RMSE: 99.4565
- Rotation Accuracy: 99.46°

### Test Set
- Average Loss: 10.2098
- Translation RMSE: 0.1059
- Translation Accuracy: 0.11 cm
- Rotation RMSE: 98.1516
- Rotation Accuracy: 98.15°

## Dataset Statistics
### Training Set
- Translation range: [-0.10, 0.34] m

### Validation Set
- Translation range: [-0.08, 0.31] m

### Test Set
- Translation range: [-0.08, 0.33] m

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch4
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/ViT6DP_batch4.3.pth
