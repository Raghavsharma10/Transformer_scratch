def val(self, piece, ref_color):
        """
        Finds value of ``Piece``

        :type: piece: Piece
        :type: ref_color: Color
        :rtype: int
        """
        if piece is None:
            return 0

        if ref_color == piece.color:
            const = 1
        else:
            const = -1

        if isinstance(piece, Pawn):
            return self.PAWN_VALUE * const
        elif isinstance(piece, Queen):
            return self.QUEEN_VALUE * const
        elif isinstance(piece, Bishop):
            return self.BISHOP_VALUE * const
        elif isinstance(piece, Rook):
            return self.ROOK_VALUE * const
        elif isinstance(piece, Knight):
            return self.KNIGHT_VALUE * const
        elif isinstance(piece, King):
            return self.KING_VALUE * const
        return 0