def half_mag_amplitude_ratio2(self, mag, avg):
        """
        Return ratio of amplitude of higher and lower magnitudes.


        A ratio of amplitude of higher and lower magnitudes than average,
        considering weights. This ratio, by definition, should be higher
        for EB than for others.

        Parameters
        ----------
        mag : array_like
            An array of magnitudes.
        avg : float
            An average value of magnitudes.

        Returns
        -------
        hl_ratio : float
            Ratio of amplitude of higher and lower magnitudes than average.
        """

        # For lower (fainter) magnitude than average.
        index = np.where(mag > avg)
        fainter_mag = mag[index]

        lower_sum = np.sum((fainter_mag - avg) ** 2) / len(fainter_mag)

        # For higher (brighter) magnitude than average.
        index = np.where(mag <= avg)
        brighter_mag = mag[index]

        higher_sum = np.sum((avg - brighter_mag) ** 2) / len(brighter_mag)

        # Return ratio.
        return np.sqrt(lower_sum / higher_sum)