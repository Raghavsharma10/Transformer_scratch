def _empty_not_in_check(self, position, direction):
        """
        Checks if set of squares in between ``King`` and ``Rook`` are empty and safe
        for the king to castle.

        :type: position: Position
        :type: direction: function
        :type: times: int
        :rtype: bool
        """
        def valid_square(square):
            return position.is_square_empty(square) and \
                   not self.in_check(position, square)

        return valid_square(direction(self.location, 1)) and \
            valid_square(direction(self.location, 2))