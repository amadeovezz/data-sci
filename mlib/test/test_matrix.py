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

    row_vector = [
        [1,2,3]
   ]

    row_vector_2 = [
        [0,8,2]
    ]

    malformed_matrix = [
        [1,2,3]
        , [1,2]
    ]

    def test_linear_combination(self):
        m = matrix.Matrix(self.column_vector)
        n = matrix.Matrix(self.column_vector_2)
        k = 3 * m + n

        assert k.matrix[0][0] == 5
        assert k.matrix[1][0] == 10
        assert k.matrix[2][0] == 15

    def test_row_vector_add(self):
        m = matrix.Matrix(self.row_vector)
        n = matrix.Matrix(self.row_vector_2)
        k = m + n

        assert k.matrix[0][0] == 1
        assert k.matrix[0][1] == 10
        assert k.matrix[0][2] == 5

    def test_column_vector_add(self):
        m = matrix.Matrix(self.column_vector)
        n = matrix.Matrix(self.column_vector_2)
        k = m + n

        assert k.matrix[0][0] == 3
        assert k.matrix[1][0] == 6
        assert k.matrix[2][0] == 9

    def test_column_vector_scaling(self):
        m = matrix.Matrix(self.column_vector)
        n = 3 * m
        assert n.matrix[0][0] == 3
        assert n.matrix[1][0] == 6
        assert n.matrix[2][0] == 9

    def test_row_vector_scaling(self):
        m = matrix.Matrix(self.row_vector)
        n = 3 * m
        assert n.matrix[0][0] == 3
        assert n.matrix[0][1] == 6
        assert n.matrix[0][2] == 9

    def test_row_vector_iteration(self):
        m = matrix.Matrix(self.row_vector)
        for element in m:
            assert type(element) == int

    def test_column_vector_iteration(self):
        m = matrix.Matrix(self.column_vector)
        for element in m:
            assert type(element) == int

    def test_transpose_row_vector_to_column(self):
        m = matrix.Matrix(self.column_vector)
        n = m.transpose()
        assert n.is_row_vector is True
        assert n.matrix[0][0] == 1
        assert n.matrix[0][1] == 2
        assert n.matrix[0][2] == 3

    def test_transpose_column_vector_to_row(self):
        m = matrix.Matrix(self.row_vector)
        n = m.transpose()
        assert n.is_column_vector is True
        assert n.matrix[0][0] == 1
        assert n.matrix[1][0] == 2
        assert n.matrix[2][0] == 3

    def test_malformed_matrix(self):
        with pytest.raises(Exception):
            matrix.Matrix(self.malformed_matrix)

    def test_column_vector_dimension(self):
        m = matrix.Matrix(self.column_vector)
        assert m.dim['row_size'] == 3
        assert m.dim['column_size'] == 1

    def test_row_vector_dimension(self):
        m = matrix.Matrix(self.row_vector)
        assert m.dim['row_size'] == 1
        assert m.dim['column_size'] == 3

    def test_is_column_vector_matrix(self):
        m = matrix.Matrix(self.column_vector)
        assert m.is_column_vector is True

    def test_is_row_vector_matrix(self):
        m = matrix.Matrix(self.row_vector)
        assert m.is_row_vector is True
