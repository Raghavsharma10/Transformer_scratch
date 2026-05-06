def move_piece(self, initial, final):
        """
        Moves piece from one location to another

        :type: initial: Location
        :type: final: Location
        """
        self.place_piece_at_square(self.piece_at_square(initial), final)
        self.remove_piece_at_square(initial)