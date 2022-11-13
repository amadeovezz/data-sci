# 3rd party
import numpy as np

# lib
from kernels import create_kernel

class TestKernels:

    def test_create_uniform_kernel(self):
        k = create_kernel(1)
        k_rounded = np.round(k, decimals=3)
        expected = np.array([
            [.111, .111, .111]
            , [.111, .111, .111]
            , [.111, .111, .111]], dtype=float)
        assert (k_rounded == expected).all()
