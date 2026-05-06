def moment1(self):
        """The first time delay weighted statistical moment of the
        instantaneous unit hydrograph."""
        delays, response = self.delay_response_series
        return statstools.calc_mean_time(delays, response)