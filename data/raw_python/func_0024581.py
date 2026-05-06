def possible_moves(self, position):
        """
        Returns all possible rook moves.

        :type: position: Board
        :rtype: list
        """
        for move in itertools.chain(*[self.moves_in_direction(fn, position) for fn in self.cross_fn]):
            yield move