# 3rd party
import pytest
import numpy as np

# lib
from convolution import kernels, convolve

class TestConvolutions:

    img = np.array([
          [1, 1, 1]
        , [2, 2, 2]
        , [1, 1, 1]
    ], dtype=float)

    img_2 = np.array([
          [1, 1, 1, 1, 1]
        , [2, 2, 2, 2, 2]
        , [3, 3, 3, 3, 3]
        , [2, 2, 2, 2, 2]
        , [1, 1, 1, 1, 1]
    ], dtype=float)

    def test_convolve_uniform_kernel(self):
        k = kernels.create_kernel(1)
        new_img = convolve.convolve(self.img, k)
        expected = np.array([
              [.667, 1, .667]
            , [.889, 1.333, .889]
            , [.667, 1, .667]], dtype=float)
        new_img_rounded = np.round(new_img, decimals=3)
        assert (new_img_rounded== expected).all()
