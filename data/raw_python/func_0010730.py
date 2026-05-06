def calc_wind_stats(self):
        """
        Calculates statistics in order to derive diurnal patterns of wind speed
        """
        a, b, t_shift = melodist.fit_cosine_function(self.data.wind)
        self.wind.update(a=a, b=b, t_shift=t_shift)