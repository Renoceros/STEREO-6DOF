# Evaluation Results - Batch 7 - Model 6

## Training Configuration
- Batch Size: 96
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
- Backbone: Using ConvNeXtV2 Nano @ 224 Image split and stacked into a 6Ch image
- Head: Linear(620 -> 256 -> 128 -> 9)

## Evaluation Metrics

### Test Set
- Average Loss: 0.7855
- Translation RMSE: 0.0583
- Translation Accuracy: 10.11 cm
- Translation Accuracy %: 81.72%
- Rotation RMSE: 44.7145
- Rotation Accuracy: 44.71°
- Rotation Accuracy % : 87.58 %
- Inference Speed: 0.95 ms/frame

### Validation Set
- Average Loss: 0.7798
- Translation RMSE: 0.0598
- Translation Accuracy: 10.38 cm
- Translation Accuracy %: 79.68%
- Rotation RMSE: 44.3662
- Rotation Accuracy: 44.37°
- Rotation Accuracy % : 87.68 %
- Inference Speed: 0.93 ms/frame

## Dataset Statistics
### Training Set
- Translation range: [-0.08, 0.53] m

### Validation Set
- Translation range: [-0.04, 0.47] m

### Test Set
- Translation range: [-0.07, 0.49] m

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch7
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/S6ch-ConvNeXt6DP7.6.pth
