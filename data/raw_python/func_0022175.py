def _create_matrix(self, byte_list):
        """
        This matrix decides which blocks should be filled fg/bg colour
        True for fg_colour
        False for bg_colour

        hash_bytes - array of hash bytes values. RGB range values in each slot

        Returns:
            List representation of the matrix
            [[True, True, True, True],
            [False, True, True, False],
            [True, True, True, True],
            [False, False, False, False]]
        """

        # Number of rows * cols halfed and rounded
        # in order to fill opposite side
        cells = int(self.rows * self.cols / 2 + self.cols % 2)

        matrix = [[False] * self.cols for num in range(self.rows)]

        for cell_number in range(cells):

            # If the bit with index corresponding to this cell is 1
            # mark that cell as fg_colour
            # Skip byte 1, that's used in determining fg_colour
            if self._bit_is_one(cell_number, byte_list[1:]):
                # Find cell coordinates in matrix.
                x_row = cell_number % self.rows
                y_col = int(cell_number / self.cols)
                # Set coord True and its opposite side
                matrix[x_row][self.cols - y_col - 1] = True
                matrix[x_row][y_col] = True
        return matrix