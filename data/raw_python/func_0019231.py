def moments(self):
        """The first two time delay weighted statistical moments of the
        ARMA response."""
        timepoints = self.ma.delays
        response = self.response
        moment1 = statstools.calc_mean_time(timepoints, response)
        moment2 = statstools.calc_mean_time_deviation(
            timepoints, response, moment1)
        return numpy.array([moment1, moment2])