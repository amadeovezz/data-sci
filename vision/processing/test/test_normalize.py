# 3rd party
import numpy as np

# lib
import normalize

class TestNormalize:

    def test_negative_normalize(self):
        """
        @return:
        """
        arr = np.array([
            [-1,-2],
            [1,2]], dtype=np.float)

        expected = np.array([
            [63, 0]
            , [191, 255]
        ])

        normalized_arr = normalize.normalize(arr)
        assert np.all(normalized_arr == expected)
        assert normalized_arr.dtype == np.uint8

    def test_positive_normalize(self):
        """
        @return:
        """
        arr = np.array([
            [1,600],
            [300,700]], dtype=np.float)

        expected = np.array([
            [0, 218]
            , [109, 255]
        ])

        normalized_arr = normalize.normalize(arr)
        assert np.all(normalized_arr == expected)
        assert normalized_arr.dtype == np.uint8
