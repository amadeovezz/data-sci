from typing import Tuple, Union
import numpy as np


def qr_factorization(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    TODO: extend for matrices of all sizes
    TODO: WIP

    Currently only factorizes 3x3 matrices.

    @param A:
    @return:
    """

    if np.linalg.det(A) == 0:
        raise Exception('Columns are not linearly independent')

    # Setup
    Q = np.zeros((3,3), dtype=float)
    R = np.zeros((3,3), dtype=float)
    a_0, a_1, a_2 = A[0,:], A[1,:], A[2,:]

    # span(a_0) = span(c * q_0)

    a_0_norm = np.linalg.norm(a_0)
    q_0 = a_0 / a_0_norm
    Q[0,:] = q_0
    R[0:0] = a_0_norm

    # span(a_0,a_1) = span(c_0 q_0, c_1 q_1)

    direction_of_q_0 = np.dot(a_1, q_0)
    projection_onto_q_0 = direction_of_q_0 * q_0
    a_1_perp = a_1 - projection_onto_q_0
    a_1_p_norm = np.linalg.norm(a_1_perp)
    q_1 = a_1_perp / a_1_p_norm
    Q[1,:] = q_1

    R[0:1] = direction_of_q_0
    R[1:1] = a_1_p_norm

    # span(a_0,a_1,a_2) = span(c_0 q_0, c_1 q_1, c_2 q_2)

    direction_of_q_0 = np.dot(a_2, q_0)
    projection_onto_q_0 = q_0 * direction_of_q_0
    direction_of_q_1 = np.dot(a_2, q_1)
    projection_onto_q_1 = q_1 * direction_of_q_1

    projection_of_a2_onto_plane = projection_onto_q_0 - projection_onto_q_1 # This can also be calculated (QQ^T)?
    a_2_perp = a_2 - projection_of_a2_onto_plane
    a_2_perp_norm = np.linalg.norm(a_2_perp)
    q_2 = a_2_perp / a_2_perp_norm
    Q[2,:] = q_2

    R[2,0] = direction_of_q_0
    R[2,1] = direction_of_q_1
    R[2,2] = a_2_perp_norm

    return (Q,R)



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

    Given by: A (A^T A)^-1 A^T  or  a (a^T a)^-1 a^T

    @param A: a matrix or a vector
    @return:
    """

    if A.ndim == 1:
        a = A
        vec = pseudo_inverse(a)
        return np.outer(vec, a)
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
