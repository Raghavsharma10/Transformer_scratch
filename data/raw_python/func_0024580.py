def moves_in_direction(self, direction, position):
        """
        Finds moves in a given direction

        :type: direction: lambda
        :type: position: Board
        :rtype: list
        """
        current_square = self.location

        while True:
            try:
                current_square = direction(current_square)
            except IndexError:
                return

            if self.contains_opposite_color_piece(current_square, position):
                yield self.create_move(current_square, notation_const.CAPTURE)

            if not position.is_square_empty(current_square):
                return

            yield self.create_move(current_square, notation_const.MOVEMENT)