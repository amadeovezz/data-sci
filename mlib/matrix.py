class Matrix:

    def __init__(self, matrix):
        self.matrix = matrix
        self.dim = self.dimension(matrix)
        self.is_column_vector = True if self.dim['column_size'] == 1 else False
        self.is_row_vector = True if self.dim['row_size'] == 1 else False

    def dimension(self, matrix) -> {}:
        """
         :param matrix:
         :return: a list with index 0 as the row dimension (m) and index 1 as the column dimension
      """
        row_size = len(matrix)
        column_size = len(matrix[0])

        # make sure each row has the same number of columns
        for i in matrix:
            if len(i) != column_size:
                raise Exception('Error! Column sizes do not match')

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
            for columns in self.matrix[0]:
                row_str += " [{column}]".format(column=columns)
            row_str += " ]"
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
