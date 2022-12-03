import numpy as np


def create_kernel(type: str, size: int = 1) -> np.ndarray:
    """
    @param type: uniform, gaussian, ...
    @param size: this is our k value. All kernels must be (2k + 1) by (2k + 1).
    Therefore the minimum size is 3x3 when k = 1.

    @return: a kernel
    """

    num_row_col = (2 * size) + 1
    # all kernels are (2 * size) + 1 by (2 * size) + 1
    if type == 'gradient_x':
        return np.array([
            [0, 0, 0]
            , [-1, 0, 1]
            , [0, 0, 0]
        ], dtype=np.float)
    elif type == 'gradient_y':
        return np.array([
            [0, -1, 0]
            , [0, 0, 0]
            , [0, 1, 0]
        ], dtype=np.float)
    elif type == 'uniform':
        normalizing_factor = 1 / (num_row_col * num_row_col)
        return normalizing_factor * np.ones((num_row_col, num_row_col), dtype=np.float)
    elif type == 'gaussian':
        # TODO: look into automatically creating these
        gaussian = None
        if size == 1:
            gaussian = np.array([
                [1, 2, 1]
                , [2, 4, 2]
                , [1, 2, 1]
            ], dtype=np.float)
        elif size == 2:
            gaussian = np.array([
                [1, 4, 7, 4, 1]
                , [4, 15, 26, 16, 4]
                , [7, 26, 41, 26, 7]
                , [4, 15, 26, 16, 4]
                , [1, 4, 7, 4, 1]
            ], dtype=np.float)
        elif size == 3:
            gaussian = np.array([
                [0, 0, 1, 2, 1, 0, 0]
                , [0, 3, 13, 22, 13, 3, 0]
                , [1, 13, 59, 97, 59, 13, 1]
                , [2, 22, 97, 157, 97, 22, 2]
                , [1, 13, 59, 97, 59, 13, 1]
                , [0, 3, 13, 22, 13, 3, 0]
                , [0, 0, 1, 2, 1, 0, 0]
            ], dtype=np.float)
        else:
            raise Exception('discrete gaussians only support kernel sizes: 3x3, 5x5, 7x7 ... ')
        normalizing_factor = gaussian.sum()
        return normalizing_factor * gaussian
    elif type == 'identity':
        return np.identity(num_row_col)
    else:
        raise Exception(f'unknown type {type}...')
