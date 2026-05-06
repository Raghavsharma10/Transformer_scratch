def play_move_msg(self, move_msg):
        """Another play move function for move message.

        Parameters
        ----------
        move_msg : string
            a valid message should be in:
            "[move type]: [X], [Y]"
        """
        move_type, move_x, move_y = self.parse_move(move_msg)
        self.play_move(move_type, move_x, move_y)