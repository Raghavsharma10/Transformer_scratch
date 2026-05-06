def check_move(self, move_type, move_x, move_y):
        """Check if a move is valid.

        If the move is not valid, then shut the game.
        If the move is valid, then setup a dictionary for the game,
        and update move counter.

        TODO: maybe instead of shut the game, can end the game or turn it into
        a valid move?

        Parameters
        ----------
        move_type : string
            one of four move types:
            "click", "flag", "unflag", "question"
        move_x : int
            X position of the move
        move_y : int
            Y position of the move
        """
        if move_type not in self.move_types:
            raise ValueError("This is not a valid move!")
        if move_x < 0 or move_x >= self.board_width:
            raise ValueError("This is not a valid X position of the move!")
        if move_y < 0 or move_y >= self.board_height:
            raise ValueError("This is not a valid Y position of the move!")

        move_des = {}
        move_des["move_type"] = move_type
        move_des["move_x"] = move_x
        move_des["move_y"] = move_y
        self.num_moves += 1

        return move_des