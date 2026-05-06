def bySignificator(self, ID):
        """ Returns all directions to a significator. """
        res = []
        for direction in self.table:
            if ID in direction[2]:
                res.append(direction)
        return res