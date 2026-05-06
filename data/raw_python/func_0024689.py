def waveset(self):
        """Optimal wavelengths for sampling the spectrum or bandpass."""
        w = get_waveset(self.model)
        if w is not None:
            utils.validate_wavelengths(w)
            w = w * self._internal_wave_unit
        return w