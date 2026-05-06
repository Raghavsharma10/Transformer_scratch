def get_SZ_single(self, alpha=None, beta=None):
        """Get the S and Z matrices for a single orientation.
        """
        if alpha == None:
            alpha = self.alpha
        if beta == None:
            beta = self.beta

        tm_outdated = self._tm_signature != (self.radius, self.radius_type, 
            self.wavelength, self.m, self.axis_ratio, self.shape, self.ddelt, 
            self.ndgs)
        if tm_outdated:
            self._init_tmatrix()

        scatter_outdated = self._scatter_signature != (self.thet0, self.thet, 
            self.phi0, self.phi, alpha, beta, self.orient)

        outdated = tm_outdated or scatter_outdated

        if outdated:
            (self._S_single, self._Z_single) = pytmatrix.calcampl(self.nmax, 
                self.wavelength, self.thet0, self.thet, self.phi0, self.phi, 
                alpha, beta)
            self._set_scatter_signature()

        return (self._S_single, self._Z_single)