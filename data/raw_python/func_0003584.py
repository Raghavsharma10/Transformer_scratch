def _id(self):
        """What this object is equal to."""
        return (self.__class__, self.number_of_needles, self.needle_positions,
                self.left_end_needle)