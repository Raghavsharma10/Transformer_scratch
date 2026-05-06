def _is_en_passant_valid(self, opponent_pawn_location, position):
        """
        Finds if their opponent's pawn is next to this pawn

        :rtype: bool
        """
        try:
            pawn = position.piece_at_square(opponent_pawn_location)
            return pawn is not None and \
                isinstance(pawn, Pawn) and \
                pawn.color != self.color and \
                position.piece_at_square(opponent_pawn_location).just_moved_two_steps
        except IndexError:
            return False