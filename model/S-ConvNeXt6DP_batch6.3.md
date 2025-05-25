# Evaluation Results - Batch 6 - Model 3

## Training Configuration
- Batch Size: 96
- Epochs: 30
- Learning Rate: 0.0001
- Translation Weight : 1.5
- Rotation Weight : 1.0
- Angular Weight : 0.1
- Patience : 3
- Image Size: 224
- Device: cuda
- Optimizer : Adam

## Model Architecture
- Backbone: Using ConvNeXtV2 Nano @ 224 Image split and stacked into a 6Ch image
- Head: Linear(620 -> 256 -> 9)

## Evaluation Metrics

### Test Set
- Average Loss: 0.2645
- Translation RMSE: 0.0500
- Translation Accuracy: 8.68 cm
- Translation Accuracy %: 84.68%
- Rotation RMSE: 14.9486
- Rotation Accuracy: 14.95°
- Rotation Accuracy % : 95.85 %
- Inference Speed: 1.13 ms/frame

### Validation Set
- Average Loss: 0.2618
- Translation RMSE: 0.0503
- Translation Accuracy: 8.73 cm
- Translation Accuracy %: 84.94%
- Rotation RMSE: 14.7909
- Rotation Accuracy: 14.79°
- Rotation Accuracy % : 95.89 %
- Inference Speed: 0.96 ms/frame

## Dataset Statistics
### Training Set
- Translation range: [-0.08, 0.53] m

### Validation Set
- Translation range: [-0.07, 0.51] m

### Test Set
- Translation range: [-0.05, 0.51] m

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch6
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/S6ch-ConvNeXt6DP6.3.pth
