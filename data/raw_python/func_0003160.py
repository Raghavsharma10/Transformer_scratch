def get_SZ_orient(self):
        """Get the S and Z matrices using the specified orientation averaging.
        """

        tm_outdated = self._tm_signature != (self.radius, self.radius_type, 
            self.wavelength, self.m, self.axis_ratio, self.shape, self.ddelt, 
            self.ndgs)
        scatter_outdated = self._scatter_signature != (self.thet0, self.thet, 
            self.phi0, self.phi, self.alpha, self.beta, self.orient)

        orient_outdated = self._orient_signature != \
            (self.orient, self.or_pdf, self.n_alpha, self.n_beta)
        if orient_outdated:
            self._init_orient()
        
        outdated = tm_outdated or scatter_outdated or orient_outdated

        if outdated:
            (self._S_orient, self._Z_orient) = self.orient(self)
            self._set_scatter_signature()

        return (self._S_orient, self._Z_orient)