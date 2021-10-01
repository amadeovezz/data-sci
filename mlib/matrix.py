class Matrix:

    def __init__(self, matrix):
        self.matrix = matrix
        self.dim = self.dimension(matrix)
        self.is_column_vector = True if self.dim['column_size'] == 1 else False
        self.is_row_vector = True if self.dim['row_size'] == 1 else False


    def transpose(self):
        """
        :return: a new instance of Matrix
        """
        if not self.is_column_vector and not self.is_row_vector:
            raise Exception('Matrix must be a column or row vector to transpose...')

        if self.is_column_vector:
            new_matrix = [[row[0] for row in self.matrix]]
            return Matrix(new_matrix)

        elif self.is_row_vector:
            new_matrix = []
            for column in self.matrix[0]:
                new_matrix.append([column])
            return Matrix(new_matrix)

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
