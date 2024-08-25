import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


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


def main():
    w_2 = np.array([[0.2, 0.9, 0.6], [0.2, 0.3, 0.7]])  # shape 2x3
    w_3 = np.array([[0.2, 0.5]])  # shape 1x2
    a_1 = np.array([[15], [5], [15]])  # shape 3x1
    y_true = np.array([[1]])  # shape 1x1
    z_2 = w_2.dot(a_1)  # shape 2x1
    a_2 = sigmoid(z_2)  # shape 2x1
    z_3 = w_3.dot(a_2)  # shape 1x1
    a_3 = sigmoid(z_3)  # shape 1x1
    delta_3 = (a_3 - y_true) * sigmoid_prime(z_3)  # shape 1x1
    delta_2 = w_3.T.dot(delta_3) * sigmoid_prime(z_2)  # shape 2x1
    print("{:.10f}".format(a_1[2, 0] * delta_2[0, 0]))
    print("{:.10f}".format(a_1[2, 0] * delta_2[1, 0]))


if __name__ == "__main__":
    main()
