import numpy as np


def get_error(
    deltas: np.ndarray,
    sums: np.ndarray,
    weights: np.ndarray,
):
    """
    compute error on the previous layer of network
    deltas - ndarray of shape (n, n_{l+1})
    sums - ndarray of shape (n, n_l)
    weights - ndarray of shape (n_{l+1}, n_l)
    """
    n = deltas.shape[0]
    return (1 / n) * np.sum((np.dot(deltas, weights) * sigmoid_prime(sums)), axis=0)
