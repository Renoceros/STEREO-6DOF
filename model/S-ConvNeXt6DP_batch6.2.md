# Evaluation Results - Batch 6 - Model 2

## Training Configuration
- Batch Size: 48
- Epochs: 25
- Learning Rate: 0.0001
- Translation Weight : 1.5
- Rotation Weight : 1.0
- Angular Weight : 0.1
- Patience : 3
- Image Size: 224
- Device: cuda
- Optimizer : Adam

## Model Architecture
- Backbone: Using ConvNeXtV2 Nano @ 224
- Head: Linear(620 -> 256 -> 9)

## Evaluation Metrics

### Test Set
- Average Loss: 0.2628
- Translation RMSE: 0.0453
- Translation Accuracy: 7.87 cm
- Translation Accuracy %: 86.11%
- Rotation RMSE: 14.8876
- Rotation Accuracy: 14.89°
- Rotation Accuracy % : 95.86 %
- Inference Speed: 1.09 ms/frame

### Validation Set
- Average Loss: 0.2510
- Translation RMSE: 0.0473
- Translation Accuracy: 8.21 cm
- Translation Accuracy %: 85.84%
- Rotation RMSE: 14.1927
- Rotation Accuracy: 14.19°
- Rotation Accuracy % : 96.06 %
- Inference Speed: 0.99 ms/frame

## Dataset Statistics
### Training Set
- Translation range: [-0.08, 0.53] m

### Validation Set
- Translation range: [-0.07, 0.51] m

### Test Set
- Translation range: [-0.05, 0.51] m

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch6
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/S6ch-ConvNeXt6DP6.2.pth
