def get_background_rms(self):
        """
        Calculate the rms of the image. The rms is calculated from the interqurtile range (IQR), to
        reduce bias from source pixels.

        Returns
        -------
        rms : float
            The image rms.

        Notes
        -----
        The rms value is cached after first calculation.
        """
        # TODO: return a proper background RMS ignoring the sources
        # This is an approximate method suggested by PaulH.
        # I have no idea where this magic 1.34896 number comes from...
        if self._rms is None:
            # Get the pixels values without the NaNs
            data = numpy.extract(self.hdu.data > -9999999, self.hdu.data)
            p25 = scipy.stats.scoreatpercentile(data, 25)
            p75 = scipy.stats.scoreatpercentile(data, 75)
            iqr = p75 - p25
            self._rms = iqr / 1.34896
        return self._rms