def place_piece_at_square(self, piece, location):
        """
        Places piece at given get_location

        :type: piece: Piece
        :type: location: Location
        """
        self.position[location.rank][location.file] = piece
        piece.location = location