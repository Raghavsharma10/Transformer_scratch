def would_move_be_promotion(self):
        """
        Finds if move from current location would be a promotion
        """
        return (self._end_loc.rank == 0 and not self.color) or \
            (self._end_loc.rank == 7 and self.color)