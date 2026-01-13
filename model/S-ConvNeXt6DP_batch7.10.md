# Evaluation Results - Batch 7 - Model 10

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
- Backbone: Using ConvNeXtV2 Nano @ 224
- Head: Linear(620 -> 512 -> 9)

## Evaluation Metrics

### Test Set
- Average Loss: 1.3837
- Translation RMSE: 0.0953
- Translation Accuracy: 16.55 cm
- Translation Accuracy %: 70.08%
- Rotation RMSE: 78.4968
- Rotation Accuracy: 78.50°
- Rotation Accuracy % : 78.20 %
- Inference Speed: 1.22 ms/frame

### Validation Set
- Average Loss: 1.3054
- Translation RMSE: 0.1009
- Translation Accuracy: 17.52 cm
- Translation Accuracy %: 65.70%
- Rotation RMSE: 73.9070
- Rotation Accuracy: 73.91°
- Rotation Accuracy % : 79.47 %
- Inference Speed: 1.12 ms/frame

## Dataset Statistics
### Training Set
- Translation range: [-0.08, 0.53] m

### Validation Set
- Translation range: [-0.04, 0.47] m

### Test Set
- Translation range: [-0.07, 0.49] m

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch7
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/S-ConvNeXt6DP7.10.pth
