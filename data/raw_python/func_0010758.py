def disaggregate_wind(self, method='equal'):
        """
        Disaggregate wind speed.

        Parameters
        ----------
        method : str, optional
            Disaggregation method.

            ``equal``
                Mean daily wind speed is duplicated for the 24 hours of the day. (Default)

            ``cosine``
                Distributes daily mean wind speed using a cosine function derived from hourly
                observations.

            ``random``
                Draws random numbers to distribute wind speed (usually not conserving the
                daily average).
        """
        self.data_disagg.wind = melodist.disaggregate_wind(self.data_daily.wind, method=method, **self.statistics.wind)