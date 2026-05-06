def calc_twi(self):
        """
        Calculates the topographic wetness index and saves the result in
        self.twi.

        Returns
        -------
        twi : array
            Array giving the topographic wetness index at each pixel
        """
        if self.uca is None:
            self.calc_uca()
            gc.collect()  # Just in case
        min_area = self.twi_min_area
        min_slope = self.twi_min_slope
        twi = self.uca.copy()
        if self.apply_twi_limits_on_uca:
            twi[twi > self.uca_saturation_limit * min_area] = \
                self.uca_saturation_limit * min_area
        gc.collect()  # Just in case
        twi = np.log((twi) / (self.mag + min_slope))
        # apply the cap
        if self.apply_twi_limits:
            twi_sat_value = \
                np.log(self.uca_saturation_limit * min_area / min_slope)
            twi[twi > twi_sat_value] = twi_sat_value
        # multiply by 10 for better integer resolution when storing
        self.twi = twi * 10

        gc.collect()  # Just in case
        return twi