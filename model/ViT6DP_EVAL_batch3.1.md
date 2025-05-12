# Evaluation Results - Batch 3

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
- Average Loss: 0.0673
- Translation RMSE: 0.0317
- Translation Accuracy: 0.03 cm
- Rotation RMSE: 0.2528
- Rotation Accuracy: 0.25°

### Test Set
- Average Loss: 0.0680
- Translation RMSE: 0.0312
- Translation Accuracy: 0.03 cm
- Rotation RMSE: 0.2532
- Rotation Accuracy: 0.25°

## Dataset Statistics
### Training Set
- Translation range: [-0.10, 0.34] m
- Rotation magnitude range: [2.43, 2.43]

### Validation Set
- Translation range: [-0.08, 0.33] m
- Rotation magnitude range: [2.43, 2.43]

### Test Set
- Translation range: [-0.08, 0.33] m
- Rotation magnitude range: [2.42, 2.42]

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch3
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/ViT6DP_batch3.1.pth
