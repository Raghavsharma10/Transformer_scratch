def fill_jacobian_column(self, jaccol, coordinates):
        """Fill in a column of the Jacobian.

           Arguments:
            | ``jaccol`` -- The column of Jacobian to which the result must be
                            added.
            | ``coordinates`` -- A numpy array with Cartesian coordinates,
                                 shape=(N,3)
        """
        q, g = self.icfn(coordinates[list(self.indexes)], 1)
        for i, j in enumerate(self.indexes):
            jaccol[3*j:3*j+3] += g[i]
        return jaccol