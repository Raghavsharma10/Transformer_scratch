def get_period_uncertainty(self, fx, fy, jmax, fx_width=100):
        """
        Get uncertainty of a period.

        The uncertainty is defined as the half width of the frequencies
        around the peak, that becomes lower than average + standard deviation
        of the power spectrum.

        Since we may not have fine resolution around the peak,
        we do not assume it is gaussian. So, no scaling factor of
        2.355 (= 2 * sqrt(2 * ln2)) is applied.

        Parameters
        ----------
        fx : array_like
            An array of frequencies.
        fy : array_like
            An array of amplitudes.
        jmax : int
            An index at the peak frequency.
        fx_width : int, optional
            Width of power spectrum to calculate uncertainty.

        Returns
        -------
        p_uncertain : float
            Period uncertainty.
        """

        # Get subset
        start_index = jmax - fx_width
        end_index = jmax + fx_width
        if start_index < 0:
            start_index = 0
        if end_index > len(fx) - 1:
            end_index = len(fx) - 1

        fx_subset = fx[start_index:end_index]
        fy_subset = fy[start_index:end_index]
        fy_mean = np.median(fy_subset)
        fy_std = np.std(fy_subset)

        # Find peak
        max_index = np.argmax(fy_subset)

        # Find list whose powers become lower than average + std.
        index = np.where(fy_subset <= fy_mean + fy_std)[0]

        # Find the edge at left and right. This is the full width.
        left_index = index[(index < max_index)]
        if len(left_index) == 0:
            left_index = 0
        else:
            left_index = left_index[-1]
        right_index = index[(index > max_index)]
        if len(right_index) == 0:
            right_index = len(fy_subset) - 1
        else:
            right_index = right_index[0]

        # We assume the half of the full width is the period uncertainty.
        half_width = (1. / fx_subset[left_index]
                      - 1. / fx_subset[right_index]) / 2.
        period_uncertainty = half_width

        return period_uncertainty