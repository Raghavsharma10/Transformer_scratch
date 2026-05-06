def dev_moments(self):
        """Sum of the absolute deviations between the central moments of the
        instantaneous unit hydrograph and the ARMA approximation."""
        return numpy.sum(numpy.abs(self.moments-self.ma.moments))