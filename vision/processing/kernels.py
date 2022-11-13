import numpy as np


def create_kernel(kernel_size: int, type: str = 'uniform') -> np.array:
    """
    @param kernel_size: this is our k value. All kernels must be (2k + 1) by (2k + 1).
    Therefore the minimum size is 3x3 when k = 1.
    @param kernel_type: uniform, ...

    @return: a kernel
    """
    # all kernels are 2 * (kernel_size) + 1 by 2 * (kernel_size) + 1
    num_row_col = (2 * kernel_size) + 1
    if type == 'uniform':
        # Get average
        value = 1/(num_row_col*num_row_col)
        return np.full((num_row_col, num_row_col),value, dtype=float)
    elif type == 'identity':
        return np.identity(num_row_col)

