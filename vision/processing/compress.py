import random
from typing import Tuple, List

import numpy as np


def low_rank_approx(img: np.ndarray, projection_columns: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs of a low rank approximation for a given set of columns.

    A is the space we are projecting on, B is our img and the approximation is given by:
    A A(A^T A)^-1 A^T B = A Y^T.

    The projection_columns must be linear independent.

    The rank k is given by len(projection_columns).

    @param img: The img to approximate
    @param projection_columns: The indexes (zero indexed) of the columns of our image we will use as our approximation.

    @return: (A, Y)
    """

    # Set-up
    m = img.shape[0]
    rank = len(projection_columns)

    A = np.zeros((m,rank), img.dtype)
    B = img

    for i in projection_columns:
        A[:, i] = img[:, i]

    # Compute
    Square = np.matmul(A.T,  A)

    # TODO: use a more efficient factorization here
    Inv = np.linalg.inv(Square)

    if rank == 1:
        # Inv is a scalar here
        BT_a = np.matmul(img.T,A)
        Y = (Inv * BT_a).T

    else:
        AT_B = np.matmul(A.T, B)
        Y = np.matmul(Inv,AT_B)

    return A, np.round(Y,3)
