def parse_move(self, move_msg):
        """Parse a move from a string.

        Parameters
        ----------
        move_msg : string
            a valid message should be in:
            "[move type]: [X], [Y]"

        Returns
        -------
        """
        # TODO: some condition check
        type_idx = move_msg.index(":")
        move_type = move_msg[:type_idx]
        pos_idx = move_msg.index(",")
        move_x = int(move_msg[type_idx+1:pos_idx])
        move_y = int(move_msg[pos_idx+1:])

        return move_type, move_x, move_y