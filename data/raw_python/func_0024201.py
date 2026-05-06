def contains_opposite_color_piece(self, square, position):
        """
        Finds if square on the board is occupied by a ``Piece``
        belonging to the opponent.

        :type: square: Location
        :type: position: Board
        :rtype: bool
        """
        return not position.is_square_empty(square) and \
            position.piece_at_square(square).color != self.color