import numpy as np
# from numba import njit


# @njit
def get_coeffs(order: int):
    # Coefficients of order r=2
    # On smooth solutions this should converge with order r=3
    Cs = []
    As = []
    Sigs = []

    Cs.append(np.array([1, 2]) / 3)
    As.append(
        np.array(
            [
                [3, -1],
                [1, 1],
            ]
        )
        / 2
    )
    Sigs.append(np.array([[[1, 0], [-2, 1]], [[1, 0], [-2, 1]]], dtype=np.float32))

    # Coefficients of order r=3
    # On smooth solutions this should converge with order r=5
    Cs.append(np.array([1, 6, 3]) / 10)
    As.append(
        np.array(
            [
                [11, -7, 2],
                [2, 5, -1],
                [-1, 5, 2],
            ]
        )
        / 6
    )
    const = 3.0
    s3 = (
        np.array(
            [
                [[10, 0, 0], [-31, 25, 0], [11, -19, 4]],
                [[4, 0, 0], [-13, 13, 0], [5, -13, 4]],
                [[4, 0, 0], [-19, 25, 0], [11, -31, 10]],
            ],
            dtype=np.float32,
        )
        / 3.0
    )
    s3 = s3.astype(np.float32)
    Sigs.append(s3)

    C = Cs[order - 2]
    a = As[order - 2]
    sigma = Sigs[order - 2]

    return C, a, sigma


# @njit
def reconstruct(order, q):
    """
    Do WENO reconstruction

    Parameters
    ----------

    order : int
        The stencil width
    q : numpy array
        Scalar data to reconstruct

    Returns
    -------

    qL : numpy array
        Reconstructed data - boundary points are zero
    """

    C, a, sigma = get_coeffs(order)

    qL = np.zeros_like(q)
    beta = np.zeros((order, len(q)))
    w = np.zeros_like(order)
    npt = len(q) - 2 * order
    epsilon = 1e-16
    for i in range(order, npt + order):
        q_stencils = np.zeros(order)
        alpha = np.zeros(order)
        for k in range(order):
            for l in range(order):
                for m in range(l + 1):
                    beta[k, i] += sigma[k, l, m] * q[i + k - l] * q[i + k - m]
            alpha[k] = C[k] / (epsilon + beta[k, i] ** 2)
            for l in range(order):
                q_stencils[k] += a[k, l] * q[i + k - l]
        w = alpha / np.sum(alpha)
        qL[i] = np.dot(w, q_stencils)

    return qL
