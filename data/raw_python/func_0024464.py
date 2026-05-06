def get_piece(self, piece_type, input_color):
        """
        Gets location of a piece on the board given the type and color.
        
        :type: piece_type: Piece
        :type: input_color: Color 
        :rtype: Location
        """
        for loc in self:
            piece = self.piece_at_square(loc)

            if not self.is_square_empty(loc) and \
                    isinstance(piece, piece_type) and \
                    piece.color == input_color:
                return loc

        raise Exception("{} \nPiece not found: {}".format(self, piece_type))