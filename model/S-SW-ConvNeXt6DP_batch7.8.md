# Evaluation Results - Batch 7 - Model 8

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
- Backbone: Using ConvNeXt V2 Nano @ 224 x2 shared weights
- Head: Linear(1280 -> 256 -> 9) (640 * 2)

## Evaluation Metrics

### Test Set
- Average Loss: 0.6675
- Translation RMSE: 0.0567
- Translation Accuracy: 9.86 cm
- Translation Accuracy %: 82.17%
- Rotation RMSE: 37.9663
- Rotation Accuracy: 37.97°
- Rotation Accuracy % : 89.45 %
- Inference Speed: 2.26 ms/frame

### Validation Set
- Average Loss: 0.6875
- Translation RMSE: 0.0577
- Translation Accuracy: 10.03 cm
- Translation Accuracy %: 80.36%
- Rotation RMSE: 39.0942
- Rotation Accuracy: 39.09°
- Rotation Accuracy % : 89.14 %
- Inference Speed: 2.06 ms/frame

## Dataset Statistics
### Training Set
- Translation range: [-0.08, 0.53] m

### Validation Set
- Translation range: [-0.04, 0.47] m

### Test Set
- Translation range: [-0.07, 0.49] m

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch7
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/S-SW-ConvNeXt6DP7.8.pth
