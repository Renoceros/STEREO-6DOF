# Evaluation Results - Batch 7 - Model 1

## Training Configuration
- Batch Size: 32
- Epochs: 20
- Learning Rate: 0.0001
- Translation Weight : 1.5
- Rotation Weight : 1.0
- Angular Weight : 0.1
- Patience : 3
- Image Size: 384
- Device: cuda
- Optimizer : Adam

## Model Architecture
- Backbone: Using ConvNeXtV2 Nano @ 384
- Head: Linear(620 -> 512 -> 9)

## Evaluation Metrics

### Test Set
- Average Loss: 0.6164
- Translation RMSE: 0.0498
- Translation Accuracy: 8.68 cm
- Translation Accuracy %: 84.30%
- Rotation RMSE: 35.1003
- Rotation Accuracy: 35.14°
- Rotation Accuracy % : 90.2387269337972 %
- Inference Speed: 0.28 ms/frame

### Validation Set
- Average Loss: 0.6002
- Translation RMSE: 0.0514
- Translation Accuracy: 8.91 cm
- Translation Accuracy %: 82.56%
- Rotation RMSE: 34.1647
- Rotation Accuracy: 34.22°
- Rotation Accuracy % : 90.49398210313585 %
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
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/S-ConvNeXt6DP7.1.pth
