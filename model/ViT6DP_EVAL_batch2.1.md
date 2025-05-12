# Evaluation Results - Batch 2

## Training Configuration
- Batch Size: 32
- Epochs: 20
- Learning Rate: 0.0001
- Image Size: 224
- Device: cuda
- Optimizer : Adam

## Model Architecture
- Backbone: ViT Base Patch16 224
- Head: Linear(768->512->6)

## Evaluation Metrics

### Validation Set
- Average Loss: 0.0587
- Translation RMSE: 0.0295
- Translation Accuracy: 0.03 cm
- Rotation RMSE: 0.2339
- Rotation Accuracy: 0.23°

### Test Set
- Average Loss: 0.0657
- Translation RMSE: 0.0293
- Translation Accuracy: 0.03 cm
- Rotation RMSE: 0.2500
- Rotation Accuracy: 0.25°

## Dataset Statistics
### Training Set
- Translation range: [-0.09, 0.34] m
- Rotation magnitude range: [2.43, 2.43]

### Validation Set
- Translation range: [-0.07, 0.33] m
- Rotation magnitude range: [2.39, 2.39]

### Test Set
- Translation range: [-0.10, 0.33] m
- Rotation magnitude range: [2.42, 2.43]

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch2
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/ViT6DP_batch2.1.pth
