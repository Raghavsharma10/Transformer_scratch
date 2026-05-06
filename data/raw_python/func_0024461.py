def _calc_all_possible_moves(self, input_color):
        """
        Returns list of all possible moves

        :type: input_color: Color
        :rtype: list
        """
        for piece in self:

            # Tests if square on the board is not empty
            if piece is not None and piece.color == input_color:

                for move in piece.possible_moves(self):

                    test = cp(self)
                    test_move = Move(end_loc=move.end_loc,
                                     piece=test.piece_at_square(move.start_loc),
                                     status=move.status,
                                     start_loc=move.start_loc,
                                     promoted_to_piece=move.promoted_to_piece)
                    test.update(test_move)

                    if self.king_loc_dict is None:
                        yield move
                        continue

                    my_king = test.piece_at_square(self.king_loc_dict[input_color])

                    if my_king is None or \
                            not isinstance(my_king, King) or \
                            my_king.color != input_color:
                        self.king_loc_dict[input_color] = test.find_king(input_color)
                        my_king = test.piece_at_square(self.king_loc_dict[input_color])

                    if not my_king.in_check(test):
                        yield move