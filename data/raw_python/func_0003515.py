def potential_cloud_pixels(self):
        """Determine potential cloud pixels (PCPs)
        Combine basic spectral testsr to get a premliminary cloud mask
        First pass, section 3.1.1 in Zhu and Woodcock 2012
        Equation 6 (Zhu and Woodcock, 2012)
        Parameters
        ----------
        ndvi: ndarray
        ndsi: ndarray
        blue: ndarray
        green: ndarray
        red: ndarray
        nir: ndarray
        swir1: ndarray
        swir2: ndarray
        cirrus: ndarray
        tirs1: ndarray
        Output
        ------
        ndarray:
            potential cloud mask, boolean
        """
        eq1 = self.basic_test()
        eq2 = self.whiteness_test()
        eq3 = self.hot_test()
        eq4 = self.nirswir_test()
        if self.sat == 'LC8':
            cir = self.cirrus_test()
            return (eq1 & eq2 & eq3 & eq4) | cir
        else:
            return eq1 & eq2 & eq3 & eq4