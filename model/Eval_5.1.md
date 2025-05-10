# Evaluation Results - Batch 5

## Training Configuration
- Batch Size: 32
- Epochs: 20
- Learning Rate: 0.0001
- Image Size: 224
- Device: cuda

## Model Architecture
- Backbone: ViT Base Patch16 224
- Head: Linear(768->512->6)

## Evaluation Metrics

### Validation Set
- Average Loss: 0.0880
- Translation RMSE: 0.0365 (normalized)
- Rotation RMSE: 0.2935 (normalized)
- Translation Accuracy: 2.74 cm
- Rotation Accuracy: 36.38°

### Test Set
- Average Loss: 0.0882
- Translation RMSE: 0.0374 (normalized)
- Rotation RMSE: 0.2934 (normalized)
- Translation Accuracy: 2.87 cm
- Rotation Accuracy: 36.48°

## File Locations
- Dataset Directory: /home/moreno/SKRIPSI/SCRIPTS/dataset/batch5
- Model Save Path: /home/moreno/SKRIPSI/SCRIPTS/model/ViT6DP_batch5.1.pth
