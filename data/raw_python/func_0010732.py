def calc_temperature_stats(self):
        """
        Calculates statistics in order to derive diurnal patterns of temperature
        """
        self.temp.max_delta = melodist.get_shift_by_data(self.data.temp, self._lon, self._lat, self._timezone)
        self.temp.mean_course = melodist.util.calculate_mean_daily_course_by_month(self.data.temp, normalize=True)