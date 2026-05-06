def disaggregate_humidity(self, method='equal', preserve_daily_mean=False):
        """
        Disaggregate relative humidity.

        Parameters
        ----------
        method : str, optional
            Disaggregation method.

            ``equal``
                Mean daily humidity is duplicated for the 24 hours of the day. (Default)

            ``minimal``:
                Calculates humidity from daily dew point temperature by setting the dew point temperature
                equal to the daily minimum temperature.

            ``dewpoint_regression``:
                Calculates humidity from daily dew point temperature by calculating dew point temperature
                using ``Tdew = a * Tmin + b``, where ``a`` and ``b`` are determined by calibration.

            ``linear_dewpoint_variation``:
                Calculates humidity from hourly dew point temperature by assuming a linear dew point
                temperature variation between consecutive days.

            ``min_max``:
                Calculates hourly humidity from observations of daily minimum and maximum humidity.

            ``month_hour_precip_mean``:
                Calculates hourly humidity from categorical [month, hour, precip(y/n)] mean values
                derived from observations.

        preserve_daily_mean : bool, optional
            If True, correct the daily mean values of the disaggregated data with the observed daily means.
        """
        self.data_disagg.hum = melodist.disaggregate_humidity(
            self.data_daily,
            temp=self.data_disagg.temp,
            method=method,
            preserve_daily_mean=preserve_daily_mean,
            **self.statistics.hum
        )