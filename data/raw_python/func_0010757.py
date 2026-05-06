def calc_sun_times(self):
        """
        Computes the times of sunrise, solar noon, and sunset for each day.
        """

        self.sun_times = melodist.util.get_sun_times(self.data_daily.index, self.lon, self.lat, self.timezone)