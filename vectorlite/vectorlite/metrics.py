import numpy as np

def l2_distance_batch(matrix: np.ndarray, q: np.ndarray) -> np.ndarray:
    # matrix: (N, D), q: (D,)
    diff = matrix - q
    return np.sum(diff * diff, axis=1)

def cosine_similarity_batch(matrix: np.ndarray, q: np.ndarray) -> np.ndarray:
    # Assumes not necessarily normalized
    q_norm = q / (np.linalg.norm(q) + 1e-12)
    mat_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return mat_norm @ q_norm
