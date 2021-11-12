# 3rd party
import pytest

# lib
from mlib import matrix


class TestMatrix:
    column_vector = [
          [1]
        , [2]
        , [3]
    ]

    column_vector_2 = [
          [2]
        , [4]
        , [6]
    ]

    column_vector_3 = [
          [2]
        , [4]
    ]

    row_vector = [
        [1,2,3]
   ]

    row_vector_2 = [
        [0,8,2]
    ]

    reflection_matrix = [
        [0, 1],
        [1, 0],
    ]

    malformed_matrix = [
        [1,2,3]
        , [1,2]
    ]

    def test_matrix_from_vector_func(self):
        def vector_func(x_0: int, x_1: int, x_2: int) -> [[]]:
            return [
                  [x_0 + 1]
                , [x_1 + 1]
                , [x_2 + 1]
            ]

        A = matrix.matrix_from_vector_func(vector_func)
        assert A.matrix[0][0] == 2
        assert A.matrix[1][0] == 1
        assert A.matrix[2][0] == 1

        assert A.matrix[0][1] == 1
        assert A.matrix[1][1] == 2
        assert A.matrix[2][1] == 1

        assert A.matrix[0][2] == 1
        assert A.matrix[1][2] == 1
        assert A.matrix[2][2] == 2

    def test_vector_matrix_multiplication(self):
        A = matrix.Matrix(self.reflection_matrix)
        v = matrix.Matrix(self.column_vector_3)
        u = A * v

        assert u.matrix[0][0] == 4
        assert u.matrix[1][0] == 2

    def test_is_matrix(self):
        m = matrix.Matrix(self.reflection_matrix)
        assert m.is_matrix is True

    def test_build_unit_basis_vectors(self):
        e_0 = matrix.new_unit_basis_vector(2, 0)
        assert e_0.matrix[0][0] == 1
        assert e_0.matrix[1][0] == 0

        e_2 = matrix.new_unit_basis_vector(3, 2)
        assert e_2.matrix[0][0] == 0
        assert e_2.matrix[1][0] == 0
        assert e_2.matrix[2][0] == 1

    def test_build_identity_matrix(self):
        I = matrix.new_identity_matrix(2)
        assert I.matrix[0][0] == 1
        assert I.matrix[1][0] == 0

        assert I.matrix[0][1] == 0
        assert I.matrix[1][1] == 1

    def test_linear_combination(self):
        v = matrix.Matrix(self.column_vector)
        u = matrix.Matrix(self.column_vector_2)
        t = 3 * v + u

        assert t.matrix[0][0] == 5
        assert t.matrix[1][0] == 10
        assert t.matrix[2][0] == 15

    def test_row_vector_add(self):
        v = matrix.Matrix(self.row_vector)
        u = matrix.Matrix(self.row_vector_2)
        t = v + u

        assert t.matrix[0][0] == 1
        assert t.matrix[0][1] == 10
        assert t.matrix[0][2] == 5

    def test_column_vector_add(self):
        v = matrix.Matrix(self.column_vector)
        u = matrix.Matrix(self.column_vector_2)
        t = v + u

        assert t.matrix[0][0] == 3
        assert t.matrix[1][0] == 6
        assert t.matrix[2][0] == 9

    def test_column_vector_scaling(self):
        v = matrix.Matrix(self.column_vector)
        u = 3 * v
        assert u.matrix[0][0] == 3
        assert u.matrix[1][0] == 6
        assert u.matrix[2][0] == 9

    def test_row_vector_scaling(self):
        v = matrix.Matrix(self.row_vector)
        u = 3 * v
        assert u.matrix[0][0] == 3
        assert u.matrix[0][1] == 6
        assert u.matrix[0][2] == 9

    def test_row_vector_iteration(self):
        v = matrix.Matrix(self.row_vector)
        for element in v:
            assert type(element) == int

    def test_column_vector_iteration(self):
        v = matrix.Matrix(self.column_vector)
        for element in v:
            assert type(element) == int

    def test_transpose_row_vector_to_column(self):
        v = matrix.Matrix(self.column_vector)
        u = v.transpose()
        assert u.is_row_vector is True
        assert u.matrix[0][0] == 1
        assert u.matrix[0][1] == 2
        assert u.matrix[0][2] == 3

    def test_transpose_column_vector_to_row(self):
        v = matrix.Matrix(self.row_vector)
        u = v.transpose()
        assert u.is_column_vector is True
        assert u.matrix[0][0] == 1
        assert u.matrix[1][0] == 2
        assert u.matrix[2][0] == 3

    def test_malformed_matrix(self):
        with pytest.raises(Exception):
            matrix.Matrix(self.malformed_matrix)

    def test_column_vector_dimension(self):
        v = matrix.Matrix(self.column_vector)
        assert v.dim['row_size'] == 3
        assert v.dim['column_size'] == 1

    def test_row_vector_dimension(self):
        v = matrix.Matrix(self.row_vector)
        assert v.dim['row_size'] == 1
        assert v.dim['column_size'] == 3

    def test_is_column_vector_matrix(self):
        v = matrix.Matrix(self.column_vector)
        assert v.is_column_vector is True

    def test_is_row_vector_matrix(self):
        v = matrix.Matrix(self.row_vector)
        assert v.is_row_vector is True
