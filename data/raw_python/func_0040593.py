def is_empty(self):
        """
        Check whether this interval is empty.

        :rtype: bool
        """
        if self.bounds[1] < self.bounds[0]:
            return True
        if self.bounds[1] == self.bounds[0]:
            return not (self.included[0] and self.included[1])