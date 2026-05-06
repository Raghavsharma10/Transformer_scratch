def temp_land(self, pcps, water):
        """Derive high/low percentiles of land temperature
        Equations 12 an 13 (Zhu and Woodcock, 2012)
        Parameters
        ----------
        pcps: ndarray
            potential cloud pixels, boolean
        water: ndarray
            water mask, boolean
        tirs1: ndarray
        Output
        ------
        tuple:
            17.5 and 82.5 percentile temperature over clearsky land
        """
        # eq 12
        clearsky_land = ~(pcps | water)

        # use clearsky_land to mask tirs1
        clear_land_temp = self.tirs1.copy()
        clear_land_temp[~clearsky_land] = np.nan
        clear_land_temp[~self.mask] = np.nan

        # take 17.5 and 82.5 percentile, eq 13
        low, high = np.nanpercentile(clear_land_temp, (17.5, 82.5))
        return low, high