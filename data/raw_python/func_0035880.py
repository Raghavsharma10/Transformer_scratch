def calcPeriod(self):
        """ calculates period using a and stellar mass
        """

        return eq.KeplersThirdLaw(self.a, self.star.M).P