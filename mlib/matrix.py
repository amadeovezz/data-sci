import logging
import math

from typing import List, Dict
from functools import reduce


class Matrix:

    def __init__(self, matrix):
        self.matrix = matrix
        self.dim = self._dimension()
        self.is_column_vector = True if self.dim['column_size'] == 1 else False
        self.is_row_vector = True if self.dim['row_size'] == 1 else False
        self.is_matrix = True if not self.is_column_vector and not self.is_row_vector else False

    def _dimension(self) -> Dict:
        """
         :return: a dictionary that contains the keys 'row_size' and 'column_size'
        """
        row_size = len(self.matrix)
        column_size = len(self.matrix[0])

        # make sure each row has the same number of columns
        for rows in self.matrix:
            if len(rows) != column_size:
                raise Exception('Error! Column sizes do not match...')

        return {'row_size': row_size, 'column_size': column_size}

    def _new_vector(self, vector: List):
        if self.is_column_vector:
            return new_column_vector(vector)
        elif self.is_row_vector:
            return new_row_vector(vector)

    @property
    def linear_combination(self) -> str:
        """
        :return: displays a linear combination of a vector
        """
        pass

    @property
    def length(self) -> float:
        # TODO: add tests for this
        if self.is_matrix:
            logging.error('Can only compute length for vectors...')
            return None

        total_sum = 0
        for element in self.vector_as_list:
            total_sum += math.sqrt(element ** 2)
        return total_sum

    @property
    def vector_as_list(self) -> List:
        """
        when matrix dimension is 1xN or Nx1, aka a (vector) then just use a list as representation, since it is
        easier to work with.
        """
        if self.is_column_vector:
            return [row[0] for row in self.matrix]
        elif self.is_row_vector:
            return [column for column in self.matrix[0]]
        else:
            return None

    def transpose(self):
        """
        :return: a new instance of Matrix
        """
        if not self.is_column_vector and not self.is_row_vector:
            raise Exception('Matrix must be a column or row vector to transpose...')

        if self.is_column_vector:
            return new_row_vector(self.vector_as_list)

        elif self.is_row_vector:
            return new_column_vector(self.vector_as_list)

        else:
            raise Exception('Can only transpose 1xN matrix or Nx1 ')

    def _vectors_op(self, operator: str, operand_left: List, operand_right: List):
        """
        helper method for adding vectors and subtracting them
        :param operator:  'sum' or 'sub'
        :param operand_left:
        :param operand_right:
        :return:
        """
        new_vector = []
        for (a, b) in zip(operand_left, operand_right):
            if operator == 'sum':
                new_vector.append(a + b)
            elif operator == 'sub':
                new_vector.append(a - b)

        return self._new_vector(new_vector)

    def __add__(self, other):
        if self.is_column_vector:
            if other.is_column_vector:
                return self._vectors_op('sum', self.vector_as_list, other)
            elif other.is_row_vector:
                return Exception('Vector shapes are incompatible... Please use transpose()...')
            else:
                return Exception('Vector and matrix shapes are incompatible...')

        elif self.is_row_vector:
            if other.is_row_vector:
                return self._vectors_op('sum', self.vector_as_list, other)
            elif other.is_column_vector:
                return Exception('Vector shapes are incompatible... Please use transpose()...')
            else:
                return Exception('Vector and matrix shapes are incompatible...')

    def __sub__(self, other):
        if self.is_column_vector:
            if other.is_column_vector:
                return self._vectors_op('sub', self.vector_as_list, other)
            elif other.is_row_vector:
                return Exception('Vector shapes are incompatible... Please use transpose()...')
            else:
                return Exception('Vector and matrix shapes are incompatible...')

        elif self.is_row_vector:
            if other.is_row_vector:
                return self._vectors_op('sub', self.vector_as_list, other)
        elif other.is_column_vector:
            return Exception('Vector shapes are incompatible... Please use transpose()...')
        else:
            return Exception('Vector and matrix shapes are incompatible...')

    def __mul__(self, other):
        # Vector matrix multiplication
        # Ex:
        """
        Ie:
        m = [
                [a_00, a_01, ..., a_0_n-1],
                [a_10, a_11, ..., a_1_n-1],
                ...
                [a_m-10, a_m-11, ..., a_m-1n-1],
            ]
        v = [
                [xo],
                [x1],
                ...
                [xi]
            ]

        m * v =
        x0 * m[0][0] + x1 * m[0][1] +  ... +
        x0 * m[1][0] + x1 * m[1][1] +  ... +
        ...
        x_i * m[i][0] + x1 * m[i][1] +  ... +

        """
        if self.is_matrix and other.is_column_vector:
            assert self.dim['column_size'] == other.dim['row_size']
            new_vector = []
            for i, rows in enumerate(self.matrix):
                total_sum = 0
                for column, scalar in zip(rows, other.vector_as_list):
                    total_sum += scalar * column
                new_vector.append(total_sum)
            return new_column_vector(new_vector)

    def __rmul__(self, other):
        if self.is_column_vector or self.is_row_vector:
            new_vector = [element * other for element in self.vector_as_list]
            return self._new_vector(new_vector)

    def __repr__(self) -> str:
        if self.is_column_vector:
            column_str = "{} x {} column vector{}".format(self.dim['row_size'], self.dim['column_size'], '\n')
            column_str += "["
            for i, rows in enumerate(self.matrix):
                if i == 0:
                    column_str += "{}    [{row}]".format('\n', row=rows[0])
                else:
                    column_str += "{}  , [{row}]".format('\n', row=rows[0])
            column_str += "{}]".format('\n')
            return column_str
        elif self.is_row_vector:
            row_str = "{} x {} row vector{}".format(self.dim['row_size'], self.dim['column_size'], '\n')
            row_str += "["
            for i, columns in enumerate(self.matrix[0]):
                if i == 0:
                    row_str += " [{column}".format(column=columns)
                else:
                    row_str += ", {column}".format(column=columns)
            row_str += "] ]"
            return row_str
        else:
            matrix_str = "{} x {} matrix{}".format(self.dim['row_size'], self.dim['column_size'], '\n')
            matrix_str += "[{}".format('\n')
            for i, rows in enumerate(self.matrix):
                for columns in rows:
                    if i == 0:
                        matrix_str += "   [{column}]".format(column=columns)
                    else:
                        matrix_str += " , [{column}]".format(column=columns)
                matrix_str += "{}".format('\n')
        matrix_str += "]"
        return matrix_str

    def __iter__(self):
        """
        Abstract away the nested list here for column and row vectors.
        :return:
        """
        if self.is_column_vector:
            for row in self.matrix:
                yield row[0]

        elif self.is_row_vector:
            for column in self.matrix[0]:
                yield column


def new_row_vector(r_vector: List) -> Matrix:
    """
    Helper method to transform a 1D list into 2D matrix object
    :param r_vector: row vector to transform
    :return:
    """
    new_matrix = [[row for row in r_vector]]
    return Matrix(new_matrix)


def new_column_vector(c_vector: List) -> Matrix:
    """
    Helper method to transform a 1D list into 2D matrix object
    :param c_vector: column vector to transform
    :return:
    """
    new_matrix = []
    for column in c_vector:
        new_matrix.append([column])
    return Matrix(new_matrix)


def new_identity_matrix(dim: int) -> Matrix:
    """
    :param dim: size of matrix -> dim x dim -> ie: 2 = 2 x 2. Minimum dimension is 2
    :return: identity matrix
    """
    if dim < 2:
        raise Exception('Dimension must be >= 2')

    # Create matrix with 0's
    identity = [[0] * dim for i in range(0, dim)]

    # Assign 1 to proper indexes
    for i, row in enumerate(identity):
        row[i] = 1
    return Matrix(identity)


def append_column_vector(target_matrix: Matrix, appending_matrix: Matrix) -> Matrix:
    """
    Appends a column vector to a given matrix.

    #TODO: make it so that we can pass in n number of columns vectors instead of re-sizing each time
    :param target_matrix:
    :param appending_matrix:
    :return:
    """
    assert appending_matrix.is_column_vector

    target_matrix_row_size = target_matrix.dim['row_size']
    target_matrix_column_size = target_matrix.dim['column_size']
    appending_matrix_row_size = appending_matrix.dim['row_size']

    if target_matrix_row_size != appending_matrix_row_size:
        raise Exception('u must have the same row dim as v...')

    # Create new matrix, with same row dim but one more column
    new_matrix = [[0] * (target_matrix_column_size + 1) for _ in range(0, target_matrix_row_size)]

    # Re-populate original matrix
    if target_matrix.is_column_vector:
        # __iter__ is defined on column and row vectors
        for i, component in enumerate(target_matrix):
            new_matrix[i][0] = component
    else:
        for i, row in enumerate(target_matrix.matrix):
            for j, component in enumerate(row):
                new_matrix[i][j] = component

    # Assign new column vector
    for i, row in enumerate(new_matrix):
        # v_column_dim is the last column (index starts at zero, otherwise it would v_column_dim + 1)
        new_matrix[i][target_matrix_column_size] = appending_matrix.matrix[i][0]

    return Matrix(new_matrix)


def new_unit_basis_vector(dim: int, unit_index: int) -> Matrix:
    """
    :param dim: size of the basis vector.
    :param unit_index: commonly seen as e_i. The index we assign our unit value 1. Index begins at 0.
    :return: unit basis column vectors
    """

    if unit_index >= dim:
        raise Exception(f"unit index must be less than dim. Vectors are indexed at 1... ")

    unit_basis_vector = [0] * dim
    unit_basis_vector[unit_index] = 1
    return new_column_vector(unit_basis_vector)


def matrix_from_vector_func(func) -> Matrix:
    """
    get_matrix transforms a generic vector function into a matrix by
    applying unit basis vectors to the function.

    :param func: user defined func
    :return:
    """

    # Infer the number of input arguments
    input_dim = func.__code__.co_argcount

    # Build unit basis vectors
    unit_basis_vectors = []
    for i in range(0, input_dim):
        unit_basis_vectors.append(new_unit_basis_vector(input_dim, i))

    # append_column_vector requires the first param to be a matrix
    # so we must create/declare the matrix outside the loop here and run the first transformation
    # TODO: refactor this
    output_vector = func(*unit_basis_vectors[0].vector_as_list)
    # Flatten since new_column_vector expects a 1D list
    transformed_unit_basis_vectors = new_column_vector(reduce(lambda x, y: x + y, output_vector))

    # Call the vector function with each component of the unit basis vectors as input (except first basis vector)
    for i in range(1, len(unit_basis_vectors)):
        output_vector = func(*unit_basis_vectors[i].vector_as_list)
        # Flatten since new_column_vector expects a 1D list
        flattened_output_vector = new_column_vector(reduce(lambda x, y: x + y, output_vector))
        transformed_unit_basis_vectors = append_column_vector(transformed_unit_basis_vectors, flattened_output_vector)

    return transformed_unit_basis_vectors


def dot(u: Matrix, v: Matrix) -> int:
    """
    #TODO: add tests
    :param u: a row or column vector
    :param v: n must be a column vector
    :return: the dot product
    """

    if not v.is_column_vector:
        raise Exception('u is not a column vector...')

    if u.vector_as_list is None and v.vector_as_list is None:
        raise Exception('v and u are not vectors...')

    if u.is_column_vector:
        logging.info('v is a column vector, transposing...')
        u = u.transpose()

    total_sum = 0
    for v_element, u_element in zip(u, v):
        total_sum += v_element * u_element
    return total_sum
