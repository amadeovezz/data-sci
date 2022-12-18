from typing import Tuple
import numpy as np


def gram_schmidt(u: np.array, v: np.array) -> Tuple[np.array, np.array]:
    """
    Creates an orthonormal basis for u and v. Assumes u is projected onto v,
    and v is kept as a basis.

    @param u: a vector
    @param v: a vector that u is projected onto (the span)
    @return:
    """

    # Normalize
    e_0 = (1/np.linalg.norm(v)) * v

    # Compute orthogonal vector to e_0
    Orth_matrix = orth_proj_matrix(e_0)
    orth = np.matmul(Orth_matrix, u)

    # Normalize
    e_1 = (1/np.linalg.norm(v)) * orth

    return (e_0, e_1)


def pseudo_inverse(A: np.ndarray, factorization: str='None') -> np.ndarray:
    """
    Computes the pseudo inverse for a given matrix A.

    Given by: (A^T A)^-1 A^T

    @param A: the matrix
    @param factorization: the type of factorization
    @return:
    """
    Square = np.matmul(A, A.T)

    # Use an efficient factorization here
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

    if len(A) == 1:
        a = A
        scalar = (1/np.dot(a,a))
        return np.outer(scalar * a, a.T)
    else:
        Pseudo = pseudo_inverse(A)
        return np.matmul(A, Pseudo)


def orth_proj_matrix(A: np.ndarray) -> np.ndarray:
    """
    Given by: I (A^T A)^-1 A^T A b
    @param A:
    @return:
    """
    Projection = proj_matrix(A)
    n = Projection.shape[1]

    return np.subtract(np.eye(n) - Projection)

def project(A: np.ndarray, b: np.ndarray) -> np.ndarray:
   """
   Project b onto the column space of A.

   (A^T A)^-1 A^T A b

   @param A:
   @param b:
   @return:
   """
   return np.matmul(proj_matrix(A), b)


