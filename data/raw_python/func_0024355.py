def on_home_row(self, location=None):
        """
        Finds out if the piece is on the home row.

        :return: bool for whether piece is on home row or not
        """
        location = location or self.location
        return (self.color == color.white and location.rank == 1) or \
               (self.color == color.black and location.rank == 6)