import numpy as np


def create_kernel(kernel_size: int, kernel_type: str = 'uniform') -> np.array:
    """
    @param kernel_size: this is our k value. All kernels must be (2k + 1) by (2k + 1).
    Therefore the minimum size is 3x3 when k = 1.
    @param kernel_type: uniform, ...

    @return: a kernel
    """
    # all kernels are 2 * (kernel_size) + 1 by 2 * (kernel_size) + 1
    kernel_dim = (2 * kernel_size) + 1
    if kernel_type == 'uniform':
        # Get average
        value = 1/(kernel_dim*kernel_dim)
        return np.full((kernel_dim, kernel_dim),value, dtype=float)

