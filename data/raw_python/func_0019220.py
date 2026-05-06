def moments(self):
        """The first two time delay weighted statistical moments of the
        MA coefficients."""
        moment1 = statstools.calc_mean_time(self.delays, self.coefs)
        moment2 = statstools.calc_mean_time_deviation(
            self.delays, self.coefs, moment1)
        return numpy.array([moment1, moment2])