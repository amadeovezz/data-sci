# 3rd party
import numpy as np

# lib
from maths.linalg import *

class TestLigAlg:

    def test_projection(self):
        a = np.array([1,1,0])
        Projection = proj_matrix(a)
        b = np.array([0,1,1])
        assert np.all(np.matmul(Projection, b) == np.array([1/2,1/2,0]))


    def test_orth_projection(self):
        a = np.array([1,1,0])
        Orth_projection = orth_proj_matrix(a)
        b = np.array([0,1,1])
        orth_vec = np.matmul(Orth_projection, b)
        assert np.dot(a, orth_vec) == 0
        assert np.all(orth_vec == np.array([-1/2,1/2,1]))


    def test_pseudo_scalar(self):
        a = np.array([1,1,0])
        scalar = pseudo_inverse(a)
        expected = np.array([1/2,1/2,0])
        assert np.all(scalar == expected)

    def test_pseudo_matrix(self):
        A = np.array([[1,0]
                     ,[0,1]
                     ,[0,1]])

        Pseudo = pseudo_inverse(A)
        Expected = np.array([[1,0,0],
                             [0,1/2,1/2]])
        assert np.all(Pseudo == Expected)


    def test_pseudo_matrix_singular(self):
        pass


    def test_gram_schmidt(self):
        u = np.array([1,1])
        v = np.array([3,-1])

        q_0, q_1 = gram_schmidt(u, v)

        assert np.round(np.linalg.norm(q_0),1) == 1
        assert np.round(np.linalg.norm(q_1,1)) == 1
        assert np.dot(q_0, q_1) == 0
        assert np.all(np.round(q_0,3) == np.array([.707, .707]))
        assert np.all(np.round(q_1,3) == np.array([.707, -.707]))

