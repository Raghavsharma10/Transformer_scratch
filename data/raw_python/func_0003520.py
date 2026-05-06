def land_temp_prob(self, tlow, thigh):
        """Temperature-based probability of cloud over land
        Equation 14 (Zhu and Woodcock, 2012)
        Parameters
        ----------
        tirs1: ndarray
        tlow: float
            Low (17.5 percentile) temperature of land
        thigh: float
            High (82.5 percentile) temperature of land
        Output
        ------
        ndarray :
            probability of cloud over land based on temperature
        """
        temp_diff = 4  # degrees
        return (thigh + temp_diff - self.tirs1) / (thigh + 4 - (tlow - 4))