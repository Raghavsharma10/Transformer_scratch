def find_difference_contour(self):
        """Find the ratio and loss/gain contours.

        This method finds the ratio contour and the Loss/Gain contour values.
        Its inputs are the two datasets for comparison where the second is the control
        to compare against the first.

        The input data sets need to be the same shape.

        Returns:
            2-element tuple containing
                - **diff** (*2D array of floats*): Ratio contour values.
                - **loss_gain_contour** (*2D array of floats*): loss/gain contour values.

        """
        # set contour to test and control contour
        self.ratio_comp_value = (self.comparison_value if self.ratio_comp_value is None
                                 else self.ratio_comp_value)

        # indices of loss,gained.
        inds_gained = np.where((self.comp1 >= self.comparison_value)
                               & (self.comp2 < self.comparison_value))
        inds_lost = np.where((self.comp1 < self.comparison_value)
                             & (self.comp2 >= self.comparison_value))

        self.comp1 = np.ma.masked_where(self.comp1 < self.ratio_comp_value, self.comp1)
        self.comp2 = np.ma.masked_where(self.comp2 < self.ratio_comp_value, self.comp2)

        # set diff to ratio for purposed of determining raito differences
        diff = self.comp1/self.comp2

        # the following determines the log10 of the ratio difference.
        # If it is extremely small, we neglect and put it as zero
        # (limits chosen to resemble ratios of less than 1.05 and greater than 0.952)
        diff = (np.log10(diff)*(diff >= 1.05)
                + (-np.log10(1.0/diff)) * (diff <= 0.952)
                + 0.0*((diff < 1.05) & (diff > 0.952)))

        # initialize loss/gain
        loss_gain_contour = np.zeros(np.shape(self.comp1))

        # fill out loss/gain
        loss_gain_contour[inds_lost] = -1
        loss_gain_contour[inds_gained] = 1

        return diff, loss_gain_contour