def calcSMAfromT(self, epsilon=0.7):
        """ Calculates the semi-major axis based on planet temperature
        """

        return eq.MeanPlanetTemp(self.albedo(), self.star.T, self.star.R, epsilon, self.T).a