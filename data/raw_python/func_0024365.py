def possible_moves(self, position):
        """
        Finds out the locations of possible moves given board.Board position.
        :pre get_location is on board and piece at specified get_location on position

        :type: position: Board
        :rtype: list
        """
        for move in itertools.chain(self.forward_moves(position),
                                    self.capture_moves(position),
                                    self.en_passant_moves(position)):
            yield move