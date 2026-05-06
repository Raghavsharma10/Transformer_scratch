def add_castle(self, position):
        """
        Adds kingside and queenside castling moves if legal

        :type: position: Board
        """
        if self.has_moved or self.in_check(position):
            return

        if self.color == color.white:
            rook_rank = 0
        else:
            rook_rank = 7

        castle_type = {
            notation_const.KING_SIDE_CASTLE: {
                "rook_file": 7,
                "direction": lambda king_square, times: king_square.shift_right(times)
            },
            notation_const.QUEEN_SIDE_CASTLE: {
                "rook_file": 0,
                "direction": lambda king_square, times: king_square.shift_left(times)
            }
        }
        for castle_key in castle_type:
            castle_dict = castle_type[castle_key]
            castle_rook = position.piece_at_square(Location(rook_rank, castle_dict["rook_file"]))
            if self._rook_legal_for_castle(castle_rook) and \
                    self._empty_not_in_check(position, castle_dict["direction"]):
                yield self.create_move(castle_dict["direction"](self.location, 2), castle_key)