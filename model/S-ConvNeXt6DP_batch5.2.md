# Evaluation Results - Batch 5 - Model 2

## Training Configuration
- Batch Size: 32
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
- Average Loss: 0.0875
- Translation RMSE: 0.0328
- Translation Accuracy: 5.71 cm
- Translation Accuracy %: 90.32%
- Rotation RMSE: 4.9576
- Rotation Accuracy: 4.96°
- Rotation Accuracy % : 98.62 %
- Inference Speed: 1.10 ms/frame

### Validation Set
- Average Loss: 0.0878
- Translation RMSE: 0.0323
- Translation Accuracy: 5.63 cm
- Translation Accuracy %: 90.30%
- Rotation RMSE: 4.9690
- Rotation Accuracy: 4.97°
- Rotation Accuracy % : 98.62 %
- Inference Speed: 0.99 ms/frame

## Dataset Statistics
### Training Set
- Translation range: [-0.08, 0.53] m

### Validation Set
- Translation range: [-0.06, 0.52] m

### Test Set
- Translation range: [-0.07, 0.52] m

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch5
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/S-ConvNeXt6DP5.2.pth
