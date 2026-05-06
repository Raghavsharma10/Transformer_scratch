def en_passant_moves(self, position):
        """
        Finds possible en passant moves.

        :rtype: list
        """

        # if pawn is not on a valid en passant get_location then return None
        if self.on_en_passant_valid_location():
            for move in itertools.chain(self.add_one_en_passant_move(lambda x: x.shift_right(), position),
                                        self.add_one_en_passant_move(lambda x: x.shift_left(), position)):
                yield move