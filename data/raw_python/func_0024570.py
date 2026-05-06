def possible_moves(self, position):
        """
        Finds all possible knight moves
        :type: position Board
        :rtype: list
        """
        for direction in [0, 1, 2, 3]:
            angles = self._rotate_direction_ninety_degrees(direction)
            for angle in angles:
                try:
                    end_loc = self.location.shift(angle).shift(direction).shift(direction)
                    if position.is_square_empty(end_loc):
                        status = notation_const.MOVEMENT
                    elif not position.piece_at_square(end_loc).color == self.color:
                        status = notation_const.CAPTURE
                    else:
                        continue

                    yield Move(end_loc=end_loc,
                               piece=self,
                               status=status,
                               start_loc=self.location)

                except IndexError:
                    pass