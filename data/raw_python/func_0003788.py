def hessian(self, coordinates):
        """Compute the force-field Hessian for the given coordinates.

           Argument:
            | ``coordinates`` -- A numpy array with the Cartesian atom
                                 coordinates, with shape (N,3).

           Returns:
            | ``hessian`` -- A numpy array with the Hessian, with shape (3*N,
                             3*N).
        """
        # N3 is 3 times the number of atoms.
        N3 = coordinates.size
        # Start with a zero hessian.
        hessian = numpy.zeros((N3,N3), float)
        # Add the contribution of each term.
        for term in self.terms:
            term.add_to_hessian(coordinates, hessian)
        return hessian