def plot(self, wavelengths=None, **kwargs):  # pragma: no cover
        """Plot the spectrum.

        .. note:: Uses ``matplotlib``.

        Parameters
        ----------
        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, `waveset` is used.

        title : str
            Plot title.

        xlog, ylog : bool
            Plot X and Y axes, respectively, in log scale.
            Default is linear scale.

        left, right : `None` or number
            Minimum and maximum wavelengths to plot.
            If `None`, uses the whole range. If a number is given,
            must be in Angstrom.

        bottom, top : `None` or number
            Minimum and maximum flux/throughput to plot.
            If `None`, uses the whole range. If a number is given,
            must be in internal unit.

        save_as : str
            Save the plot to an image file. The file type is
            automatically determined by given file extension.

        Raises
        ------
        synphot.exceptions.SynphotError
            Invalid inputs.

        """
        w, y = self._get_arrays(wavelengths)
        self._do_plot(w, y, **kwargs)