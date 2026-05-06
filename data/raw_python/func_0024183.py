def init_manual(cls, pawn_value, knight_value, bishop_value, rook_value, queen_value, king_value):
        """
        Manual init method for external piece values

        :type: PAWN_VALUE: int
        :type: KNIGHT_VALUE: int
        :type: BISHOP_VALUE: int
        :type: ROOK_VALUE: int
        :type: QUEEN_VALUE: int
        """
        piece_values = cls()
        piece_values.PAWN_VALUE = pawn_value
        piece_values.KNIGHT_VALUE = knight_value
        piece_values.BISHOP_VALUE = bishop_value
        piece_values.ROOK_VALUE = rook_value
        piece_values.QUEEN_VALUE = queen_value
        piece_values.KING_VALUE = king_value
        return piece_values