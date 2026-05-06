def capture_moves(self, position):
        """
        Finds out all possible capture moves

        :rtype: list
        """
        try:
            right_diagonal = self.square_in_front(self.location.shift_right())
            for move in self._one_diagonal_capture_square(right_diagonal, position):
                yield move
        except IndexError:
            pass

        try:
            left_diagonal = self.square_in_front(self.location.shift_left())
            for move in self._one_diagonal_capture_square(left_diagonal, position):
                yield move
        except IndexError:
            pass