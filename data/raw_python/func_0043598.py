def shift(self, count=1):
        """
        Shift the view a specified number of times.

        :param count: The count of times to shift the view.
        """
        if self:
            self._index = (self._index + count) % len(self)
        else:
            self._index = 0