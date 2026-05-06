def view(self, arcmin, arcmax):
        """ Returns the directions within the
        min and max arcs.

        """
        res = []
        for direction in self.table:
            if arcmin < direction[0] < arcmax:
                res.append(direction)
        return res