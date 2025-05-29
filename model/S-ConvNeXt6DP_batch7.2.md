# Evaluation Results - Batch 7 - Model 2

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
- Backbone: Using ConvNeXtV2 Nano @ 224
- Head: Linear(620 -> 256 -> 9)

## Evaluation Metrics

### Test Set
- Average Loss: 0.6791
- Translation RMSE: 0.0568
- Translation Accuracy: 9.87 cm
- Translation Accuracy %: 82.16%
- Rotation RMSE: 38.6326
- Rotation Accuracy: 38.62°
- Rotation Accuracy % : 89.27170647515192 %
- Inference Speed: 0.21 ms/frame

### Validation Set
- Average Loss: 0.6945
- Translation RMSE: 0.0559
- Translation Accuracy: 9.70 cm
- Translation Accuracy %: 81.00%
- Rotation RMSE: 39.5278
- Rotation Accuracy: 39.53°
- Rotation Accuracy % : 89.01866594950359 %
- Inference Speed: 0.19 ms/frame

## Dataset Statistics
### Training Set
- Translation range: [-0.08, 0.53] m

### Validation Set
- Translation range: [-0.04, 0.47] m

### Test Set
- Translation range: [-0.07, 0.49] m

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch7
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/S-ConvNeXt6DP7.2.pth
