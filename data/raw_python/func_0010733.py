def calc_radiation_stats(self, data_daily=None, day_length=None, how='all'):
        """
        Calculates statistics in order to derive solar radiation from sunshine duration or
        minimum/maximum temperature.

        Parameters
        ----------
        data_daily : DataFrame, optional
            Daily data from the associated ``Station`` object.

        day_length : Series, optional
            Day lengths as calculated by ``calc_sun_times``.
        """
        assert how in ('all', 'seasonal', 'monthly')

        self.glob.mean_course = melodist.util.calculate_mean_daily_course_by_month(self.data.glob)

        if data_daily is not None:
            pot_rad = melodist.potential_radiation(
                melodist.util.hourly_index(data_daily.index),
                self._lon, self._lat, self._timezone)
            pot_rad_daily = pot_rad.resample('D').mean()
            obs_rad_daily = self.data.glob.resample('D').mean()

            if how == 'all':
                month_ranges = [np.arange(12) + 1]
            elif how == 'seasonal':
                month_ranges = [[3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 1, 2]]
            elif how == 'monthly':
                month_ranges = zip(np.arange(12) + 1)

            def myisin(s, v):
                return pd.Series(s).isin(v).values

            def extract_months(s, months):
                return s[myisin(s.index.month, months)]

            if 'ssd' in data_daily and day_length is not None:
                for months in month_ranges:
                    a, b = melodist.fit_angstroem_params(
                        extract_months(data_daily.ssd, months),
                        extract_months(day_length, months),
                        extract_months(pot_rad_daily, months),
                        extract_months(obs_rad_daily, months),
                    )

                    for month in months:
                        self.glob.angstroem.loc[month] = a, b

            if 'tmin' in data_daily and 'tmax' in data_daily:
                df = pd.DataFrame(
                    data=dict(
                        tmin=data_daily.tmin,
                        tmax=data_daily.tmax,
                        pot_rad=pot_rad_daily,
                        obs_rad=obs_rad_daily,
                    )
                ).dropna(how='any')

                for months in month_ranges:
                    a, c = melodist.fit_bristow_campbell_params(
                        extract_months(df.tmin, months),
                        extract_months(df.tmax, months),
                        extract_months(df.pot_rad, months),
                        extract_months(df.obs_rad, months),
                    )

                    for month in months:
                        self.glob.bristcamp.loc[month] = a, c