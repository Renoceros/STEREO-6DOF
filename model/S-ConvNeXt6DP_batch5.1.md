# Evaluation Results - Batch 5 - Model 1

## Training Configuration
- Batch Size: 32
- Epochs: 20
- Learning Rate: 0.0001
- Translation Weight : 1.5
- Rotation Weight : 1.0
- Angular Weight : 0.1
- Patience : 3
- Image Size: 384
- Device: cuda
- Optimizer : Adam

## Model Architecture
- Backbone: Using ConvNeXt V2 Nano @ 384
- Head: Linear(768->512->9)

## Evaluation Metrics

### Test Set
- Average Loss: 0.3168
- Translation RMSE: 0.0007
- Translation Accuracy: 8.05 cm
- Translation Accuracy %: 86.36%
- Rotation RMSE: 0.2642
- Rotation Accuracy: 17.97°
- Inference Speed: 0.15 ms/frame

### Validation Set
- Average Loss: 0.3149
- Translation RMSE: 0.0461
- Translation Accuracy: 8.01 cm
- Translation Accuracy %: 86.20%
- Rotation RMSE: 17.8586
- Rotation Accuracy: 17.86°
- Inference Speed: 0.15 ms/frame

## Dataset Statistics
### Training Set
- Translation range: [-0.08, 0.53] m

### Validation Set
- Translation range: [-0.06, 0.52] m

### Test Set
- Translation range: [-0.07, 0.52] m

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch5
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/S-ConvNeXt6DP5.2.pth
