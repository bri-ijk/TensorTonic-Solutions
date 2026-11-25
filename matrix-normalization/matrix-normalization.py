import numpy as np


def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    # Write code here
    m = np.asarray(matrix, dtype=float)

    if norm_type == "l1":
        n = np.sum(np.abs(m), axis, keepdims=True)
    elif norm_type == "l2":
        n = np.sqrt(np.sum(m**2, axis, keepdims=True))
    elif norm_type == "max":
        n = np.max(np.abs(m), axis, keepdims=True)
    else:
        raise Exception == "sorry"

    # remove zeros
    n = np.where(n == 0, 1.0, n)
    return m / n
