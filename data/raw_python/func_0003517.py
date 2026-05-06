def water_temp_prob(self):
        """Temperature probability for water
        Equation 9 (Zhu and Woodcock, 2012)
        Parameters
        ----------
        water_temp: float
            82.5th percentile temperature over water
        swir2: ndarray
        tirs1: ndarray
        Output
        ------
        ndarray:
            probability of cloud over water based on temperature
        """
        temp_const = 4.0  # degrees C
        water_temp = self.temp_water()
        return (water_temp - self.tirs1) / temp_const