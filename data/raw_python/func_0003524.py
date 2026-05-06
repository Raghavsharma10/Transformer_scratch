def potential_snow_layer(self):
        """Spectral test to determine potential snow
        Uses the 9.85C (283K) threshold defined in Zhu, Woodcock 2015
        Parameters
        ----------
        ndsi: ndarray
        green: ndarray
        nir: ndarray
        tirs1: ndarray
        Output
        ------
        ndarray:
            boolean, True is potential snow
        """
        return (self.ndsi > 0.15) & (self.tirs1 < 9.85) & (self.nir > 0.11) & (self.green > 0.1)