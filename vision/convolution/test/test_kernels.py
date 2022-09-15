# 3rd party
import numpy as np

# lib
from convolution import kernels

class TestKernels:


    def test_create_uniform_kernel(self):
        k = kernels.create_kernel(1)
        k_rounded = np.round(k, decimals=3)
        expected = np.array([
            [.111, .111, .111]
            , [.111, .111, .111]
            , [.111, .111, .111]], dtype=float)
        assert (k_rounded == expected).all()
