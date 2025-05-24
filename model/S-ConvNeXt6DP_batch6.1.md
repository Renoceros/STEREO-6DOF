# Evaluation Results - Batch 6 - Model 1

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
- Backbone: Using ConvNeXtV2 Nano @ 224
- Head: Linear(620 -> 256 -> 9)

## Evaluation Metrics

### Test Set
- Average Loss: 0.2765
- Translation RMSE: 0.0521
- Translation Accuracy: 9.03 cm
- Translation Accuracy %: 84.07%
- Rotation RMSE: 15.6071
- Rotation Accuracy: 15.61°
- Rotation Accuracy % : 95.66 %
- Inference Speed: 0.94 ms/frame

### Validation Set
- Average Loss: 0.2677
- Translation RMSE: 0.0524
- Translation Accuracy: 9.08 cm
- Translation Accuracy %: 84.33%
- Rotation RMSE: 15.1130
- Rotation Accuracy: 15.11°
- Rotation Accuracy % : 95.80 %
- Inference Speed: 0.94 ms/frame

## Dataset Statistics
### Training Set
- Translation range: [-0.08, 0.53] m

### Validation Set
- Translation range: [-0.07, 0.51] m

### Test Set
- Translation range: [-0.05, 0.51] m

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch6
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/S-ConvNeXt6DP6.1.pth
