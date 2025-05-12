# Evaluation Results - Batch 1

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
- Average Loss: 0.1960
- Translation RMSE: 0.0453
- Translation Accuracy: 0.05 cm
- Rotation RMSE: 0.4351
- Rotation Accuracy: 0.44°

### Test Set
- Average Loss: 0.1793
- Translation RMSE: 0.0452
- Translation Accuracy: 0.05 cm
- Rotation RMSE: 0.4147
- Rotation Accuracy: 0.41°

## Dataset Statistics
### Training Set
- Translation range: [-0.08, 0.33] m
- Rotation magnitude range: [2.43, 2.43]

### Validation Set
- Translation range: [-0.07, 0.31] m
- Rotation magnitude range: [2.38, 2.38]

### Test Set
- Translation range: [-0.09, 0.32] m
- Rotation magnitude range: [2.39, 2.35]

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch1
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/ViT6DP_batch1.1.pth
