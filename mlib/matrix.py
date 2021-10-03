from typing import List


class Matrix:

    def __init__(self, matrix):
        self.matrix = matrix
        self.dim = self.dimension(matrix)
        self.is_column_vector = True if self.dim['column_size'] == 1 else False
        self.is_row_vector = True if self.dim['row_size'] == 1 else False
        self.vector = self._get_vector() if self.is_column_vector or self.is_row_vector else None

    def transpose(self):
        """
        :return: a new instance of Matrix
        """
        if not self.is_column_vector and not self.is_row_vector:
            raise Exception('Matrix must be a column or row vector to transpose...')

        if self.is_column_vector:
            return self._new_row_vector(self.vector)

        elif self.is_row_vector:
            return self._new_column_vector(self.vector)

        else:
            raise Exception('Can only transpose 1xN matrix or Nx1 ')


    def dimension(self, matrix) -> {}:
        """
         :param matrix: represented as a nested list
         :return: a dictionary that contains the keys 'row_size' and 'column_size'
        """
        row_size = len(matrix)
        column_size = len(matrix[0])

        # make sure each row has the same number of columns
        for rows in matrix:
            if len(rows) != column_size:
                raise Exception('Error! Column sizes do not match...')

        return {'row_size': row_size, 'column_size': column_size}


    def _get_vector(self) -> List:
        '''
        when matrix dimension is 1xN or Nx1, aka a (vector) then just use a list as representation, since it is
        easier to work with.
        '''
        if self.is_column_vector:
            return [row[0] for row in self.matrix]
        elif self.is_row_vector:
            return [column for column in self.matrix[0]]


    def _new_vector(self, vector: List):
        if self.is_column_vector:
            return self._new_column_vector(vector)
        elif self.is_row_vector:
            return self._new_row_vector(vector)

    @staticmethod
    def _new_column_vector(c_vector: List):
        new_matrix = []
        for column in c_vector:
            new_matrix.append([column])
        return Matrix(new_matrix)

    @staticmethod
    def _new_row_vector(r_vector: List):
        new_matrix = [[row for row in r_vector]]
        return Matrix(new_matrix)

    def __add__(self, other):
        if self.is_column_vector:
            if other.is_column_vector:
                new_vector = []
                for (a, b) in zip(self.vector, other):
                    new_vector.append(a + b)
                return self._new_vector(new_vector)
            elif other.is_row_vector:
                return Exception('Vector shapes are incompatible... Please use transpose()...')
            else:
                return Exception('Vector and matrix shapes are incompatible...')

        elif self.is_row_vector:
            if other.is_row_vector:
                new_vector = []
                for (a, b) in zip(self.vector, other):
                    new_vector.append(a + b)
                return self._new_vector(new_vector)
            elif other.is_column_vector:
                return Exception('Vector shapes are incompatible... Please use transpose()...')
            else:
                return Exception('Vector and matrix shapes are incompatible...')

    def __rmul__(self, other):
        if self.is_column_vector or self.is_row_vector:
            new_vector = [element*other for element in self.vector]
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
        '''
        Abstract away the nested list here for column and row vectors
        :return:
        '''
        if self.is_column_vector:
            for row in self.matrix:
                yield row[0]

        elif self.is_row_vector:
            for column in self.matrix[0]:
                yield column