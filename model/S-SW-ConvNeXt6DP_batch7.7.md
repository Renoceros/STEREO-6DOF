# Evaluation Results - Batch 7 - Model 7

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
- Backbone: Using ConvNeXt V2 Nano @ 224 x2 shared weights
- Head: Linear(1280 -> 256 -> 9) (640 * 2)

## Evaluation Metrics

### Test Set
- Average Loss: 2.3865
- Translation RMSE: 0.3149
- Translation Accuracy: 54.64 cm
- Translation Accuracy %: 1.19%
- Rotation RMSE: 128.1804
- Rotation Accuracy: 128.18°
- Rotation Accuracy % : 64.39 %
- Inference Speed: 2.33 ms/frame

### Validation Set
- Average Loss: 2.3960
- Translation RMSE: 0.3158
- Translation Accuracy: 54.83 cm
- Translation Accuracy %: 0.00%
- Rotation RMSE: 128.6735
- Rotation Accuracy: 128.67°
- Rotation Accuracy % : 64.26 %
- Inference Speed: 1.97 ms/frame

## Dataset Statistics
### Training Set
- Translation range: [-0.08, 0.53] m

### Validation Set
- Translation range: [-0.04, 0.47] m

### Test Set
- Translation range: [-0.07, 0.49] m

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch7
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/S-SW-ConvNeXt6DP7.7.pth
