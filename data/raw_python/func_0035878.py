def calcSMA(self):
        """ Calculates the semi-major axis from Keplers Third Law
        """
        try:
            return eq.KeplersThirdLaw(None, self.star.M, self.P).a
        except HierarchyError:
            return np.nan