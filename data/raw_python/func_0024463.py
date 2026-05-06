def find_piece(self, piece):
        """
        Finds Location of the first piece that matches piece.
        If none is found, Exception is raised.

        :type: piece: Piece
        :rtype: Location
        """
        for i, _ in enumerate(self.position):
            for j, _ in enumerate(self.position):
                loc = Location(i, j)

                if not self.is_square_empty(loc) and \
                        self.piece_at_square(loc) == piece:
                    return loc

        raise ValueError("{} \nPiece not found: {}".format(self, piece))