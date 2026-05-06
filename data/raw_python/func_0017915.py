def slope_percentile(self, date, mag):
        """
        Return 10% and 90% percentile of slope.

        Parameters
        ----------
        date : array_like
            An array of phase-folded date. Sorted.
        mag : array_like
            An array of phase-folded magnitudes. Sorted by date.

        Returns
        -------
        per_10 : float
            10% percentile values of slope.
        per_90 : float
            90% percentile values of slope.
        """

        date_diff = date[1:] - date[:len(date) - 1]
        mag_diff = mag[1:] - mag[:len(mag) - 1]

        # Remove zero mag_diff.
        index = np.where(mag_diff != 0.)
        date_diff = date_diff[index]
        mag_diff = mag_diff[index]

        # Derive slope.
        slope = date_diff / mag_diff

        percentile_10 = np.percentile(slope, 10.)
        percentile_90 = np.percentile(slope, 90.)

        return percentile_10, percentile_90