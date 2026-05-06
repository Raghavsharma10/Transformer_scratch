def byPromissor(self, ID):
        """ Returns all directions to a promissor. """
        res = []
        for direction in self.table:
            if ID in direction[1]:
                res.append(direction)
        return res