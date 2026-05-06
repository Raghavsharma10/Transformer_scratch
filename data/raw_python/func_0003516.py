def temp_water(self):
        """Use water to mask tirs and find 82.5 pctile
        Equation 7 and 8 (Zhu and Woodcock, 2012)
        Parameters
        ----------
        is_water: ndarray, boolean
            water mask, water is True, land is False
        swir2: ndarray
        tirs1: ndarray
        Output
        ------
        float:
            82.5th percentile temperature over water
        """
        # eq7
        th_swir2 = 0.03
        water = self.water_test()
        clear_sky_water = water & (self.swir2 < th_swir2)

        # eq8
        clear_water_temp = self.tirs1.copy()
        clear_water_temp[~clear_sky_water] = np.nan
        clear_water_temp[~self.mask] = np.nan
        pctl_clwt = np.nanpercentile(clear_water_temp, 82.5)
        return pctl_clwt