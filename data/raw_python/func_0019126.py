def moment2(self):
        """The second time delay weighted statistical momens of the
        instantaneous unit hydrograph."""
        moment1 = self.moment1
        delays, response = self.delay_response_series
        return statstools.calc_mean_time_deviation(
            delays, response, moment1)