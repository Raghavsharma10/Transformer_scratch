def variability_prob(self, whiteness):
        """Use the probability of the spectral variability
        to identify clouds over land.
        Equation 15 (Zhu and Woodcock, 2012)
        Parameters
        ----------
        ndvi: ndarray
        ndsi: ndarray
        whiteness: ndarray
        Output
        ------
        ndarray :
            probability of cloud over land based on variability
        """

        if self.sat in ['LT5', 'LE7']:
            # check for green and red saturation

            # if red is saturated and less than nir, ndvi = LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF
            mod_ndvi = np.where(self.red_saturated & (self.nir > self.red), 0, self.ndvi)

            # if green is saturated and less than swir1, ndsi = LE07_clip_L1TP_039027_20150529_20160902_01_T1_B1.TIF
            mod_ndsi = np.where(self.green_saturated & (self.swir1 > self.green), 0, self.ndsi)
            ndi_max = np.fmax(np.absolute(mod_ndvi), np.absolute(mod_ndsi))

        else:
            ndi_max = np.fmax(np.absolute(self.ndvi), np.absolute(self.ndsi))

        f_max = 1.0 - np.fmax(ndi_max, whiteness)

        return f_max