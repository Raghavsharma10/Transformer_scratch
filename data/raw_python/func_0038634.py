def values(self, start=None, stop=None):
        """Like :meth:`items` but returns only the values."""
        return (item[1] for item in self.items(start, stop))