def _init_tmatrix(self):
        """Initialize the T-matrix.
        """

        if self.radius_type == Scatterer.RADIUS_MAXIMUM:
            # Maximum radius is not directly supported in the original
            # so we convert it to equal volume radius
            radius_type = Scatterer.RADIUS_EQUAL_VOLUME
            radius = self.equal_volume_from_maximum()
        else:
            radius_type = self.radius_type
            radius = self.radius

        self.nmax = pytmatrix.calctmat(radius, radius_type,
            self.wavelength, self.m.real, self.m.imag, self.axis_ratio,
            self.shape, self.ddelt, self.ndgs)
        self._tm_signature = (self.radius, self.radius_type, self.wavelength,
            self.m, self.axis_ratio, self.shape, self.ddelt, self.ndgs)