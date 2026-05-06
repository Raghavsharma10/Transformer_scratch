def play_move(self, move_type, move_x, move_y):
        """Updat board by a given move.

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
        # record the move
        if self.game_status == 2:
            self.move_history.append(self.check_move(move_type, move_x,
                                                     move_y))
        else:
            self.end_game()

        # play the move, update the board
        if move_type == "click":
            self.board.click_field(move_x, move_y)
        elif move_type == "flag":
            self.board.flag_field(move_x, move_y)
        elif move_type == "unflag":
            self.board.unflag_field(move_x, move_y)
        elif move_type == "question":
            self.board.question_field(move_x, move_y)

        # check the status, see if end the game
        if self.board.check_board() == 0:
            self.game_status = 0  # game loses
            # self.print_board()
            self.end_game()
        elif self.board.check_board() == 1:
            self.game_status = 1  # game wins
            # self.print_board()
            self.end_game()
        elif self.board.check_board() == 2:
            self.game_status = 2