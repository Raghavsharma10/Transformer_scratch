def possible_moves(self, position):
        """
        Generates list of possible moves

        :type: position: Board
        :rtype: list
        """
        # Chain used to combine multiple generators
        for move in itertools.chain(*[self.add(fn, position) for fn in self.cardinal_directions]):
            yield move

        for move in self.add_castle(position):
            yield move