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

    row_vector = [
        [1,2,3]
   ]

    malformed_matrix = [
        [1,2,3]
        , [1,2]
    ]


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
