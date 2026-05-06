def as_spectrum(self, binned=True, wavelengths=None):
        """Reduce the observation to an empirical source spectrum.

        An observation is a complex object with some restrictions
        on its capabilities. At times, it would be useful to work
        with the observation as a simple object that is easier to
        manipulate and takes up less memory.

        This is also useful for writing an observation as sampled
        spectrum out to a FITS file.

        Parameters
        ----------
        binned : bool
            Write out data in native wavelengths if `False`.
            Else, write binned data (default).

        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` or `binset` is used, depending
            on ``binned``.

        Returns
        -------
        sp : `~synphot.spectrum.SourceSpectrum`
            Empirical source spectrum.

        """
        if binned:
            w, y = self._get_binned_arrays(
                wavelengths, self._internal_flux_unit)
        else:
            w, y = self._get_arrays(
                wavelengths, flux_unit=self._internal_flux_unit)

        header = {'observation': str(self), 'binned': binned}
        return SourceSpectrum(Empirical1D, points=w, lookup_table=y,
                              meta={'header': header})