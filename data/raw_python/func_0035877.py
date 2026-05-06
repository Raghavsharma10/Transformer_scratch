def calcTemperature(self):
        """ Calculates the temperature using which uses equations.MeanPlanetTemp, albedo assumption and potentially
        equations.starTemperature.

        issues
        - you cant get the albedo assumption without temp but you need it to calculate the temp.
        """
        try:
            return eq.MeanPlanetTemp(self.albedo, self.star.T, self.star.R, self.a).T_p
        except (ValueError, HierarchyError):  # ie missing value (.a) returning nan
            return np.nan