def calc_humidity_stats(self):
        """
        Calculates statistics in order to derive diurnal patterns of relative humidity.
        """
        a1, a0 = melodist.calculate_dewpoint_regression(self.data, return_stats=False)
        self.hum.update(a0=a0, a1=a1)
        self.hum.kr = 12

        self.hum.month_hour_precip_mean = melodist.calculate_month_hour_precip_mean(self.data)