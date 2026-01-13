# Evaluation Results - Batch 7 - Model 9

## Training Configuration
- Batch Size: 48
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
- Head: Linear(620 -> 512 -> 9)

## Evaluation Metrics

### Test Set
- Average Loss: 0.7000
- Translation RMSE: 0.0544
- Translation Accuracy: 9.43 cm
- Translation Accuracy %: 82.94%
- Rotation RMSE: 39.8550
- Rotation Accuracy: 39.86°
- Rotation Accuracy % : 88.93 %
- Inference Speed: 1.25 ms/frame

### Validation Set
- Average Loss: 0.7084
- Translation RMSE: 0.0565
- Translation Accuracy: 9.82 cm
- Translation Accuracy %: 80.77%
- Rotation RMSE: 40.3080
- Rotation Accuracy: 40.31°
- Rotation Accuracy % : 88.80 %
- Inference Speed: 1.11 ms/frame

## Dataset Statistics
### Training Set
- Translation range: [-0.08, 0.53] m

### Validation Set
- Translation range: [-0.04, 0.47] m

### Test Set
- Translation range: [-0.07, 0.49] m

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch7
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/S6ch-ConvNeXt6DP7.9.pth
