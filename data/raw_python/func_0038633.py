def keys(self, start=None, stop=None):
        """Like :meth:`items` but returns only the keys."""
        return (item[0] for item in self.items(start, stop))