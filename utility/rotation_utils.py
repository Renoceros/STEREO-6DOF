import numpy as np

def rotation_matrix_to_6d(R: np.ndarray) -> np.ndarray:
    """
    Converts a 3x3 rotation matrix into a 6D representation (first 2 columns).
    Used in 6D continuous representation from Zhou et al.
    """
    return R[:, :2].reshape(-1)