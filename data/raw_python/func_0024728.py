def _get_binned_arrays(self, wavelengths, flux_unit, area=None,
                           vegaspec=None):
        """Get binned observation in user units."""
        x = self._validate_binned_wavelengths(wavelengths)
        y = self.sample_binned(wavelengths=x, flux_unit=flux_unit, area=area,
                               vegaspec=vegaspec)

        if isinstance(wavelengths, u.Quantity):
            w = x.to(wavelengths.unit, u.spectral())
        else:
            w = x

        return w, y