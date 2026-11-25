import numpy as np


def batch_generator(
    X: list, y: list, batch_size: int, seed: int = 42, drop_last: bool = False
):
    """
    Returns a generator of (X_batch, y_batch) tuples.
    """
    # Write code here
    X, y = np.asarray(X), np.asarray(y)
    idx = np.arange(len(X))
    np.random.default_rng(seed).shuffle(idx)

    for i in range(0, len(idx), batch_size):
        batch_range = idx[i : i + batch_size]
        if drop_last == True and batch_range.size < batch_size:
            break
        yield X[batch_range], y[batch_range]
