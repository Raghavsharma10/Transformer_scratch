def taper(self, wavelengths=None):
        """Taper the spectrum or bandpass.

        The wavelengths to use for the first and last points are
        calculated by using the same ratio as for the 2 interior points.

        Parameters
        ----------
        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for tapering.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, `waveset` is used.

        Returns
        -------
        sp : `BaseSpectrum`
            Tapered empirical spectrum or bandpass.
            ``self`` is returned if already tapered (e.g., box model).

        """
        x = self._validate_wavelengths(wavelengths)

        # Calculate new end points for tapering
        w1 = x[0] ** 2 / x[1]
        w2 = x[-1] ** 2 / x[-2]

        # Special handling for empirical data.
        # This is to be compatible with ASTROLIB PYSYNPHOT behavior.
        if isinstance(self._model, Empirical1D):
            y1 = self._model.lookup_table[0]
            y2 = self._model.lookup_table[-1]
        # Other models can just evaluate at new end points
        else:
            y1 = self(w1)
            y2 = self(w2)

        # Nothing to do
        if y1 == 0 and y2 == 0:
            return self  # Do we need a deepcopy here?

        y = self(x)

        if y1 != 0:
            x = np.insert(x, 0, w1)
            y = np.insert(y, 0, 0.0)
        if y2 != 0:
            x = np.insert(x, x.size, w2)
            y = np.insert(y, y.size, 0.0)

        return self.__class__(Empirical1D, points=x, lookup_table=y)