def _validate_wavelengths(self, wave):
        """Validate wavelengths for sampling."""
        if wave is None:
            if self.waveset is None:
                raise exceptions.SynphotError(
                    'self.waveset is undefined; '
                    'Provide wavelengths for sampling.')
            wavelengths = self.waveset
        else:
            w = self._process_wave_param(wave)
            utils.validate_wavelengths(w)
            wavelengths = w * self._internal_wave_unit

        return wavelengths