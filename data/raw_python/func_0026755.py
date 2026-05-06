def _check(self):
        """Check that entry attributes are legal."""
        # Run the super method
        super(Photometry, self)._check()

        err_str = None
        has_flux = self._KEYS.FLUX in self
        has_flux_dens = self._KEYS.FLUX_DENSITY in self
        has_u_flux = self._KEYS.U_FLUX in self
        has_u_flux_dens = self._KEYS.U_FLUX_DENSITY in self

        has_freq = self._KEYS.FREQUENCY in self
        has_band = self._KEYS.BAND in self
        has_ener = self._KEYS.ENERGY in self
        has_u_freq = self._KEYS.U_FREQUENCY in self
        has_u_ener = self._KEYS.U_ENERGY in self

        if has_flux or has_flux_dens:
            if not any([has_freq, has_band, has_ener]):
                err_str = ("Has `{}` or `{}`".format(self._KEYS.FLUX,
                                                     self._KEYS.FLUX_DENSITY) +
                           " but None of `{}`, `{}`, `{}`".format(
                               self._KEYS.FREQUENCY, self._KEYS.BAND,
                               self._KEYS.ENERGY))
            elif has_flux and not has_u_flux:
                err_str = "`{}` provided without `{}`.".format(
                    self._KEYS.FLUX, self._KEYS.U_FLUX)
            elif has_flux_dens and not has_u_flux_dens:
                err_str = "`{}` provided without `{}`.".format(
                    self._KEYS.FLUX_DENSITY, self._KEYS.U_FLUX_DENSITY)
            elif has_freq and not has_u_freq:
                err_str = "`{}` provided without `{}`.".format(
                    self._KEYS.FREQUENCY, self._KEYS.U_FREQUENCY)
            elif has_ener and not has_u_ener:
                err_str = "`{}` provided without `{}`.".format(
                    self._KEYS.ENERGY, self._KEYS.U_ENERGY)

        if err_str is not None:
            raise ValueError(err_str)

        return