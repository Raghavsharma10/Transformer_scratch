def interpolate(self, column_hours, method='linear', limit=24, limit_direction='both', **kwargs):
        """
        Wrapper function for ``pandas.Series.interpolate`` that can be used to
        "disaggregate" values using various interpolation methods.

        Parameters
        ----------
        column_hours : dict
            Dictionary containing column names in ``data_daily`` and the hour
            values they should be associated to.

        method, limit, limit_direction, **kwargs
            These parameters are passed on to ``pandas.Series.interpolate``.

        Examples
        --------
        Assume that ``mystation.data_daily.T7``, ``mystation.data_daily.T14``,
        and ``mystation.data_daily.T19`` contain air temperature measurements
        taken at 07:00, 14:00, and 19:00.
        We can use the interpolation functions provided by pandas/scipy to derive
        hourly values:

        >>> mystation.data_hourly.temp = mystation.interpolate({'T7': 7, 'T14': 14, 'T19': 19}) # linear interpolation (default)
        >>> mystation.data_hourly.temp = mystation.interpolate({'T7': 7, 'T14': 14, 'T19': 19}, method='cubic') # cubic spline
        """
        kwargs = dict(kwargs, method=method, limit=limit, limit_direction=limit_direction)
        data = melodist.util.prepare_interpolation_data(self.data_daily, column_hours)
        return data.interpolate(**kwargs)