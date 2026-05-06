def _get_arrays(self, wavelengths, **kwargs):
        """Get sampled spectrum or bandpass in user units."""
        x = self._validate_wavelengths(wavelengths)
        y = self(x, **kwargs)

        if isinstance(wavelengths, u.Quantity):
            w = x.to(wavelengths.unit, u.spectral())
        else:
            w = x

        return w, y