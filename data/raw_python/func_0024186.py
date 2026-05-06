def white_move(self):
        """
        Calls the white player's ``generate_move()``
        method and updates the board with the move returned.
        """
        move = self.player_white.generate_move(self.position)
        move = make_legal(move, self.position)
        self.position.update(move)