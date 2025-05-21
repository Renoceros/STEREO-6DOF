# Evaluation Results - Batch 2 - Model 1

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
- Average Loss: 11.4909
- Translation RMSE: 0.1890
- Translation Accuracy: 0.19 cm
- Rotation RMSE: 108.4415
- Rotation Accuracy: 108.44°

### Test Set
- Average Loss: 11.3900
- Translation RMSE: 0.1902
- Translation Accuracy: 0.19 cm
- Rotation RMSE: 107.4032
- Rotation Accuracy: 107.40°

## Dataset Statistics
### Training Set
- Translation range: [-0.09, 0.34] m

### Validation Set
- Translation range: [-0.07, 0.33] m

### Test Set
- Translation range: [-0.10, 0.33] m

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch2
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/ViT6DP_batch2.1.pth
