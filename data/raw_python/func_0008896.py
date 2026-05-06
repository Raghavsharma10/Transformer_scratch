def get(self, key, side):
        """
        Returns an edge given a particular key
        Parmeters
        ----------
        key : tuple
            (te, be, le, re) tuple that identifies a tile
        side : str
            top, bottom, left, or right, which edge to return
        """
        return getattr(self, side).ravel()[self.keys[key]]