# Evaluation Results - Batch 7 - Model 3

## Training Configuration
- Batch Size: 64
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
- Average Loss: 0.7799
- Translation RMSE: 0.0632
- Translation Accuracy: 10.97 cm
- Translation Accuracy %: 80.16%
- Rotation RMSE: 44.3430
- Rotation Accuracy: 44.05°
- Rotation Accuracy % : 87.76 %
- Inference Speed: 0.17 ms/frame

### Validation Set
- Average Loss: 0.7643
- Translation RMSE: 0.0643
- Translation Accuracy: 10.86 cm
- Translation Accuracy %: 78.74%
- Rotation RMSE: 43.4311
- Rotation Accuracy: 43.59°
- Rotation Accuracy % : 87.89 %
- Inference Speed: 0.15 ms/frame

## Dataset Statistics
### Training Set
- Translation range: [-0.08, 0.53] m

### Validation Set
- Translation range: [-0.04, 0.47] m

### Test Set
- Translation range: [-0.07, 0.49] m

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch7
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/S-ConvNeXt6DP7.3.pth
