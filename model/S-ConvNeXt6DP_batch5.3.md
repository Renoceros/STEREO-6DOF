# Evaluation Results - Batch 5 - Model 3

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
- Head: Linear(620 -> 512 -> 9)

## Evaluation Metrics

### Test Set
- Average Loss: 0.3419
- Translation RMSE: 0.0578
- Translation Accuracy: 10.03 cm
- Translation Accuracy %: 83.00%
- Rotation RMSE: 19.3009
- Rotation Accuracy: 19.30°
- Rotation Accuracy % : 94.64 %
- Inference Speed: 0.95 ms/frame

### Validation Set
- Average Loss: 0.3127
- Translation RMSE: 0.0571
- Translation Accuracy: 9.90 cm
- Translation Accuracy %: 82.93%
- Rotation RMSE: 17.6401
- Rotation Accuracy: 17.64°
- Rotation Accuracy % : 95.10 %
- Inference Speed: 0.94 ms/frame

## Dataset Statistics
### Training Set
- Translation range: [-0.08, 0.53] m

### Validation Set
- Translation range: [-0.06, 0.52] m

### Test Set
- Translation range: [-0.07, 0.52] m

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch5
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/S-ConvNeXt6DP5.3.pth
