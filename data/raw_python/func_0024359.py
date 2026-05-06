def _one_diagonal_capture_square(self, capture_square, position):
        """
        Adds specified diagonal as a capture move if it is one
        """
        if self.contains_opposite_color_piece(capture_square, position):

            if self.would_move_be_promotion():
                for move in self.create_promotion_moves(status=notation_const.CAPTURE_AND_PROMOTE,
                                                        location=capture_square):
                    yield move

            else:
                yield self.create_move(end_loc=capture_square,
                                       status=notation_const.CAPTURE)