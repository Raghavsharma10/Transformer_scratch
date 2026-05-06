def generate_move(self, position):
        """
        Returns valid and legal move given position

        :type: position: Board
        :rtype: Move
        """
        while True:
            print(position)
            raw = input(str(self.color) + "\'s move \n")
            move = converter.short_alg(raw, self.color, position)

            if move is None:
                continue

            return move