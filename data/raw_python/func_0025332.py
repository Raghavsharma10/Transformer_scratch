def update(self, a, b, c, d):
        """
        Update contingency table with new values without creating a new object.
        """
        self.table.ravel()[:] = [a, b, c, d]
        self.N = self.table.sum()