def in_check(self, position, location=None):
        """
        Finds if the king is in check or if both kings are touching.

        :type: position: Board
        :return: bool
        """
        location = location or self.location
        for piece in position:

            if piece is not None and piece.color != self.color:
                if not isinstance(piece, King):
                    for move in piece.possible_moves(position):

                        if move.end_loc == location:
                            return True
                else:
                    if self.loc_adjacent_to_opponent_king(piece.location, position):
                        return True

        return False