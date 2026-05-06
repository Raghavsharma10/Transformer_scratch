def would_move_be_promotion(self, location=None):
        """
        Finds if move from current get_location would result in promotion

        :type: location: Location
        :rtype: bool
        """
        location = location or self.location
        return (location.rank == 1 and self.color == color.black) or \
                (location.rank == 6 and self.color == color.white)