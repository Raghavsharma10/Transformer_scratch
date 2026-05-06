def evaluate(self, x, temperature):
        """Evaluate the model.

        Parameters
        ----------
        x : number or ndarray
            Wavelengths in Angstrom.

        temperature : number
            Temperature in Kelvin.

        Returns
        -------
        y : number or ndarray
            Blackbody radiation in PHOTLAM.

        """
        bbflux = super(BlackBodyNorm1D, self).evaluate(x, temperature)
        return bbflux * self._omega