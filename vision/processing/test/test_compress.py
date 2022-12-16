# 3rd party
import numpy as np

# lib
from compress import low_rank_approx


class TestCompress:

    def test_rank_1_approx(self):
        img = np.array([
              [.9,1.1,1,4]
            , [1,1.1,1.1,4.1]
            , [1,1.2,1.3,4.2]
        ])

        A, L = low_rank_approx(img, [0])
        assert np.all(np.matmul(A,L)[:,0] == img[:,0])

    def test_rank_2_approx(self):
        img = np.array([
            [.9,1.1,1,4]
            , [1,1.1,1.1,4.1]
            , [1,1.2,1.3,4.2]
        ])

        A, L = low_rank_approx(img, [0,1])
        assert np.all(np.matmul(A,L)[:,0] == img[:,0])
        assert np.all(np.matmul(A,L)[:,1] == img[:,1])


