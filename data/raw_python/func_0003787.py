def add_to_hessian(self, coordinates, hessian):
        """Add the contributions of this energy term to the Hessian

           Arguments:
            | ``coordinates`` -- A numpy array with 3N Cartesian coordinates.
            | ``hessian`` -- A matrix for the full Hessian to which this energy
                             term has to add its contribution.
        """
        # Compute the derivatives of the bond stretch towards the two cartesian
        # coordinates. The bond length is computed too, but not used.
        q, g = self.icfn(coordinates[list(self.indexes)], 1)
        # Add the contribution to the Hessian (an outer product)
        for ja, ia in enumerate(self.indexes):
            # ja is 0, 1, 2, ...
            # ia is i0, i1, i2, ...
            for jb, ib in enumerate(self.indexes):
                contrib = 2*self.force_constant*numpy.outer(g[ja], g[jb])
                hessian[3*ia:3*ia+3, 3*ib:3*ib+3] += contrib