def in_check_as_result(self, pos, move):
        """
        Finds if playing my move would make both kings meet.

        :type: pos: Board
        :type: move: Move
        :rtype: bool
        """
        test = cp(pos)
        test.update(move)
        test_king = test.get_king(move.color)

        return self.loc_adjacent_to_opponent_king(test_king.location, test)