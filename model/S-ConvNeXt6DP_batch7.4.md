# Evaluation Results - Batch 7 - Model 4

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
- Head: Linear(620 -> 256 -> 128-> 9)

## Evaluation Metrics

### Test Set
- Average Loss: 1.9467
- Translation RMSE: 0.0749
- Translation Accuracy: 13.07 cm
- Translation Accuracy %: 76.37%
- Rotation RMSE: 111.0467
- Rotation Accuracy: 110.95°
- Rotation Accuracy % : 69.18 %
- Inference Speed: 0.16 ms/frame

### Validation Set
- Average Loss: 1.9143
- Translation RMSE: 0.0750
- Translation Accuracy: 12.99 cm
- Translation Accuracy %: 74.56%
- Rotation RMSE: 109.1739
- Rotation Accuracy: 109.56°
- Rotation Accuracy % : 69.57 %
- Inference Speed: 0.17 ms/frame

## Dataset Statistics
### Training Set
- Translation range: [-0.08, 0.53] m

### Validation Set
- Translation range: [-0.04, 0.47] m

### Test Set
- Translation range: [-0.07, 0.49] m

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch7
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/S-ConvNeXt6DP7.4.pth
