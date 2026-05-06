def z(self, what):
        """Change redshift."""
        if not isinstance(what, numbers.Real):
            raise exceptions.SynphotError(
                'Redshift must be a real scalar number.')
        self._z = float(what)
        self._redshift_model = RedshiftScaleFactor(self._z)
        if self.z_type == 'wavelength_only':
            self._redshift_flux_model = None
        else:  # conserve_flux
            self._redshift_flux_model = Scale(1 / (1 + self._z))