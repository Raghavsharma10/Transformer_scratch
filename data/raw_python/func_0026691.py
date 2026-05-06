def _check(self):
        """Check that spectrum has legal combination of attributes."""
        # Run the super method
        super(Spectrum, self)._check()

        err_str = None
        has_data = self._KEYS.DATA in self
        has_wave = self._KEYS.WAVELENGTHS in self
        has_flux = self._KEYS.FLUXES in self
        has_filename = self._KEYS.FILENAME in self

        if not has_data:
            if (not has_wave or not has_flux) and not has_filename:
                err_str = (
                    "If `{}` not given".format(self._KEYS.DATA) +
                    "; `{}` or `{}` needed".format(
                        self._KEYS.WAVELENGTHS, self._KEYS.FLUXES))

        if err_str is not None:
            raise ValueError(err_str)

        return