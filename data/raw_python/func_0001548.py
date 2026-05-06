def copy(self):
        """Return a shallow copy of a RangeSet."""
        cpy = self.__class__()
        cpy._autostep = self._autostep
        cpy.padding = self.padding
        cpy.update(self)
        return cpy