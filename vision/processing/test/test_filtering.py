# 3rd party
import pytest
import numpy as np

# lib
from filtering import filter
from kernels import create_kernel

class TestFiltering:

    simple_img = np.array([
          [1, 2, 3]
        , [2, 3, 4]
        , [5, 6, 7]
    ], dtype=float)

    img = np.array([
          [1, 1, 1]
        , [2, 2, 2]
        , [1, 1, 1]
    ], dtype=float)

    img_3 = np.array([
          [1, 1, 1, 1, 1]
        , [2, 2, 2, 2, 2]
        , [3, 3, 3, 3, 3]
        , [2, 2, 2, 2, 2]
        , [1, 1, 1, 1, 1]
    ], dtype=float)

    def test_correlation(self):
        asymmetric_kernel = np.array([
            [-1, -1, -1]
            , [1, 1, 1]
            , [1, 1, 1]
        ], dtype=float)

        new_img = filter(self.simple_img, asymmetric_kernel)
        assert new_img[1,1] == 21

    def test_convolution(self):
        asymmetric_kernel = np.array([
            [-1, -1, -1]
            , [1, 1, 1]
            , [1, 1, 1]
        ], dtype=float)

        new_img = filter(self.simple_img, asymmetric_kernel, convolve=True)
        assert new_img[1,1] == -3.0

    def test_filter_uniform_kernel(self):
        k = create_kernel('uniform')
        new_img = filter(self.img, k)
        expected = np.array([
              [.667, 1, .667]
            , [.889, 1.333, .889]
            , [.667, 1, .667]], dtype=float)
        new_img_rounded = np.round(new_img, decimals=3)
        assert (new_img_rounded == expected).all()
