def _process_wave_param(self, pval):
        """Process individual model parameter representing wavelength."""
        return self._process_generic_param(
            pval, self._internal_wave_unit, equivalencies=u.spectral())