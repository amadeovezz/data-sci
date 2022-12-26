from typing import Tuple, Union
import numpy as np


def find_inverse(A: np.ndarray) -> Tuple[np.array]:
    pass



def qr_factorization(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pass


def gram_schmidt(u: np.array, v: np.array) -> Tuple[np.array, np.array]:
    """
    Creates an orthonormal basis for u and v. Assumes v is projected onto u.

    @param u: a vector
    @param v: a vector that u is projected onto (the span)
    @return:
    """

    # Normalize
    q_0 = (1/np.linalg.norm(u)) * u

    # Compute orthogonal vector to q_0
    Orth_matrix = orth_proj_matrix(u)
    orth_vect = np.matmul(Orth_matrix, v)

    # Normalize
    q_1 = (1/np.linalg.norm(orth_vect)) * orth_vect

    return (q_0, q_1)


def pseudo_inverse(A: np.ndarray, factorization: str='None') -> Union[float, np.ndarray]:
    """
    Computes the pseudo inverse for a given non-square matrix A or a vector a.

    Given by: (A^T A)^-1 A^T or for vectors (a^T a)^-1 a^T

    @param A: the matrix
    @param factorization: the type of factorization
    @return:
    """

    # Do we have a vector
    if A.ndim == 1:
        a = A
        scalar = (1/np.dot(a,a))
        return scalar * a
    # Do we have a matrix
    else:
        # Is it singular
        Square = np.matmul(A.T, A)
        if np.linalg.det(Square) == 0:
            raise NotImplemented('Singular pseudo-inverse not implemented')

        else:
            Inv = np.linalg.inv(Square)
            return np.matmul(Inv, A.T)


def proj_matrix(A: np.ndarray) -> np.ndarray:
    """
    Computes a projection matrix that projects any vector b onto the column space of
    of A. A can also be a vector, in which case a vector b is projected onto the span of A.

    In either case the algebra is almost identical.

    Given by: A (A^T A)^-1 A^T  or  a a(a^T a)^-1 a^T

    @param A: a matrix or a vector
    @return:
    """

    if A.ndim == 1:
        a = A
        scalar = pseudo_inverse(a)
        return np.outer(scalar * a, a.T)
    else:
        Pseudo = pseudo_inverse(A)
        return np.matmul(A, Pseudo)


def orth_proj_matrix(A: np.ndarray) -> np.ndarray:
    """
    Given by: I - (A^T A)^-1 A^T A
    @param A:
    @return:
    """
    Projection = proj_matrix(A)
    n = Projection.shape[1]

    return np.subtract(np.eye(n), Projection)
