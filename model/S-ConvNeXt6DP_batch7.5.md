# Evaluation Results - Batch 7 - Model 5

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
- Average Loss: 0.6761
- Translation RMSE: 0.0561
- Translation Accuracy: 9.76 cm
- Translation Accuracy %: 82.36%
- Rotation RMSE: 38.4632
- Rotation Accuracy: 38.55°
- Rotation Accuracy % : 89.29 %
- Inference Speed: 0.28 ms/frame

### Validation Set
- Average Loss: 0.6997
- Translation RMSE: 0.0590
- Translation Accuracy: 10.22 cm
- Translation Accuracy %: 80.00%
- Rotation RMSE: 39.7865
- Rotation Accuracy: 39.82°
- Rotation Accuracy % : 88.94 %
- Inference Speed: 0.18 ms/frame

## Dataset Statistics
### Training Set
- Translation range: [-0.08, 0.53] m

### Validation Set
- Translation range: [-0.04, 0.47] m

### Test Set
- Translation range: [-0.07, 0.49] m

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch7
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/S-ConvNeXt6DP7.5.pth
