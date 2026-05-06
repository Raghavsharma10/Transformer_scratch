def get_corner(self, time):
        """
        Gets the corner array indices of the STObject at a given time that corresponds 
        to the upper left corner of the bounding box for the STObject.

        Args:
            time: time at which the corner is being extracted.

        Returns:
              corner index.
        """
        if self.start_time <= time <= self.end_time:
            diff = time - self.start_time
            return self.i[diff][0, 0], self.j[diff][0, 0]
        else:
            return -1, -1