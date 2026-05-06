def on_en_passant_valid_location(self):
        """
        Finds out if pawn is on enemy center rank.

        :rtype: bool
        """
        return (self.color == color.white and self.location.rank == 4) or \
               (self.color == color.black and self.location.rank == 3)