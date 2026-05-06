def half_mag_amplitude_ratio(self, mag, avg, weight):
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
        weight : array_like
            An array of weight.

        Returns
        -------
        hl_ratio : float
            Ratio of amplitude of higher and lower magnitudes than average.
        """

        # For lower (fainter) magnitude than average.
        index = np.where(mag > avg)
        lower_weight = weight[index]
        lower_weight_sum = np.sum(lower_weight)
        lower_mag = mag[index]
        lower_weighted_std = np.sum((lower_mag
                                     - avg) ** 2 * lower_weight) / \
                             lower_weight_sum

        # For higher (brighter) magnitude than average.
        index = np.where(mag <= avg)
        higher_weight = weight[index]
        higher_weight_sum = np.sum(higher_weight)
        higher_mag = mag[index]
        higher_weighted_std = np.sum((higher_mag
                                      - avg) ** 2 * higher_weight) / \
                              higher_weight_sum

        # Return ratio.
        return np.sqrt(lower_weighted_std / higher_weighted_std)