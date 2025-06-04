import numpy as np
import torch
import torch.nn.functional as F

# --- Loss and Metric Functions ---
def rotation_error(R_pred, R_gt):
    """Compute angular error in degrees between rotation matrices."""
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = torch.diagonal(R_diff, dim1=1, dim2=2).sum(dim=1)
    eps = 1e-6
    angle_rad = torch.acos(torch.clamp((trace - 1) / 2, min=-1 + eps, max=1 - eps))
    return torch.rad2deg(angle_rad)

def geodesic_loss(R_pred, R_gt):
    """Computes the geodesic loss between two rotation matrices."""
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = torch.diagonal(R_diff, dim1=1, dim2=2).sum(dim=1)
    eps = 1e-6
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1 + eps, 1 - eps))
    return angle.mean()

def compute_rotation_matrix_from_ortho6d(poses_6d):
    """Convert 6D rotation representation to 3x3 rotation matrices."""
    x_raw = poses_6d[:, 0:3]
    y_raw = poses_6d[:, 3:6]
    x = F.normalize(x_raw, dim=1)
    z = F.normalize(torch.cross(x, y_raw, dim=1), dim=1)
    y = torch.cross(z, x, dim=1)
    rot = torch.stack((x, y, z), dim=-1)  # Shape: [B, 3, 3]
    return rot

def combined_loss(output, target, trans_w=1.0, rot_w=1.0): # Removed angular_w as it wasn't used in your original example
    """
    Calculates a combined loss of translation (MSE) and rotation (geodesic).
    """
    pred_trans = output[:, :3]
    gt_trans = target[:, :3]
    pred_rot_6d = output[:, 3:9]
    gt_rot_6d = target[:, 3:9]

    pred_rot = compute_rotation_matrix_from_ortho6d(pred_rot_6d)
    gt_rot = compute_rotation_matrix_from_ortho6d(gt_rot_6d)

    loss_trans = F.mse_loss(pred_trans, gt_trans)
    loss_rot = geodesic_loss(pred_rot, gt_rot)
    return trans_w * loss_trans + rot_w * loss_rot

def rotation_error_deg_from_6d(pred_6d, gt_6d):
    """
    Computes the mean angular error in degrees from 6D rotation representations.
    """
    pred_6d = pred_6d.float()
    gt_6d = gt_6d.float()
    R_pred = compute_rotation_matrix_from_ortho6d(pred_6d)
    R_gt = compute_rotation_matrix_from_ortho6d(gt_6d)
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    cos_theta = (trace - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)
    return torch.rad2deg(theta.mean())

def compute_errors(outputs, labels):
    """
    Computes translation RMSE and rotation RMSE from model outputs and ground truth labels.
    """
    outputs = outputs.float()
    labels = labels.float()
    pred_trans = outputs[:, :3]
    pred_rot_6d = outputs[:, 3:9]
    gt_trans = labels[:, :3]
    gt_rot_6d = labels[:, 3:9]

    trans_rmse = torch.sqrt(F.mse_loss(pred_trans, gt_trans))
    rot_rmse = rotation_error_deg_from_6d(pred_rot_6d, gt_rot_6d)
    return trans_rmse.item(), rot_rmse.item()

def calculate_translation_rmse(preds, gts):
    """Calculates translation RMSE in centimeters."""
    trans_rmse = np.sqrt(np.mean(np.sum((preds[:, :3] - gts[:, :3])**2, axis=1)))
    return trans_rmse * 100  # Convert m → cm

def translation_accuracy_percentage(rmse_cm, range_cm):
    """Calculates translation accuracy percentage."""
    return max(0.0, 100.0 * (1 - rmse_cm / range_cm))

def get_dataset_stats(loader):
    """Calculates translation statistics (min, max, mean, std) for a dataset."""
    translations = []
    for batch_data in loader:
        # Handle different dataset outputs
        if isinstance(batch_data, tuple) and len(batch_data) == 3: # Stereo or 6ch
            _, _, labels = batch_data
        else: # Vanilla
            _, labels = batch_data
        translations.append(labels[:, :3])
    trans = torch.cat(translations, dim=0)
    return {
        'min': trans.min(dim=0).values,
        'max': trans.max(dim=0).values,
        'mean': trans.mean(dim=0),
        'std': trans.std(dim=0)
    }

def safe(val):
    """Safely converts numpy floats/integers to Python native types."""
    if isinstance(val, (np.floating, np.integer)):
        return val.item()
    return val
