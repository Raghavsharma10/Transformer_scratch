def evaluate(self, x, *args):
        """One dimensional constant flux model function.

        Parameters
        ----------
        x : number or ndarray
            Wavelengths in Angstrom.

        Returns
        -------
        y : number or ndarray
            Flux in PHOTLAM.

        """
        a = (self.amplitude * np.ones_like(x)) * self._flux_unit
        y = units.convert_flux(x, a, units.PHOTLAM)
        return y.value