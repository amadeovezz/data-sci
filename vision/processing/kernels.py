import numpy as np


def create_kernel(type: str, size: int=1) -> np.ndarray:
    """
    @param size: this is our k value. All kernels must be (2k + 1) by (2k + 1).
    Therefore the minimum size is 3x3 when k = 1.
    @param kernel_type: uniform, ...

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
        # Get average
        value = 1/(num_row_col*num_row_col)
        return np.full( (num_row_col, num_row_col), value, dtype=np.float)
    elif type == 'identity':
        return np.identity(num_row_col)


