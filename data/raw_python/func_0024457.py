def init_default(cls):
        """
        Creates a ``Board`` with the standard chess starting position.

        :rtype: Board
        """
        return cls([

            # First rank
            [Rook(white, Location(0, 0)), Knight(white, Location(0, 1)), Bishop(white, Location(0, 2)),
             Queen(white, Location(0, 3)), King(white, Location(0, 4)), Bishop(white, Location(0, 5)),
             Knight(white, Location(0, 6)), Rook(white, Location(0, 7))],

            # Second rank
            [Pawn(white, Location(1, file)) for file in range(8)],

            # Third rank
            [None for _ in range(8)],

            # Fourth rank
            [None for _ in range(8)],

            # Fifth rank
            [None for _ in range(8)],

            # Sixth rank
            [None for _ in range(8)],

            # Seventh rank
            [Pawn(black, Location(6, file)) for file in range(8)],

            # Eighth rank
            [Rook(black, Location(7, 0)), Knight(black, Location(7, 1)), Bishop(black, Location(7, 2)),
             Queen(black, Location(7, 3)), King(black, Location(7, 4)), Bishop(black, Location(7, 5)),
             Knight(black, Location(7, 6)), Rook(black, Location(7, 7))]
        ])