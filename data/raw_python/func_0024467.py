def update(self, move):
        """
        Updates position by applying selected move

        :type: move: Move
        """
        if move is None:
            raise TypeError("Move cannot be type None")

        if self.king_loc_dict is not None and isinstance(move.piece, King):
            self.king_loc_dict[move.color] = move.end_loc

        # Invalidates en-passant
        for square in self:
            pawn = square
            if isinstance(pawn, Pawn):
                pawn.just_moved_two_steps = False

        # Sets King and Rook has_moved property to True is piece has moved
        if type(move.piece) is King or type(move.piece) is Rook:
            move.piece.has_moved = True

        elif move.status == notation_const.MOVEMENT and \
                isinstance(move.piece, Pawn) and \
                fabs(move.end_loc.rank - move.start_loc.rank) == 2:
            move.piece.just_moved_two_steps = True

        if move.status == notation_const.KING_SIDE_CASTLE:
            self.move_piece(Location(move.end_loc.rank, 7), Location(move.end_loc.rank, 5))
            self.piece_at_square(Location(move.end_loc.rank, 5)).has_moved = True

        elif move.status == notation_const.QUEEN_SIDE_CASTLE:
            self.move_piece(Location(move.end_loc.rank, 0), Location(move.end_loc.rank, 3))
            self.piece_at_square(Location(move.end_loc.rank, 3)).has_moved = True

        elif move.status == notation_const.EN_PASSANT:
            self.remove_piece_at_square(Location(move.start_loc.rank, move.end_loc.file))

        elif move.status == notation_const.PROMOTE or \
                move.status == notation_const.CAPTURE_AND_PROMOTE:
            try:
                self.remove_piece_at_square(move.start_loc)
                self.place_piece_at_square(move.promoted_to_piece(move.color, move.end_loc), move.end_loc)
            except TypeError as e:
                raise ValueError("Promoted to piece cannot be None in Move {}\n{}".format(repr(move), e))
            return

        self.move_piece(move.piece.location, move.end_loc)