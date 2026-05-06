def _parse_met(content):
        """Parse standard meteorological data from NDBC buoys.

        Parameters
        ----------
        content : str
            Data to parse

        Returns
        -------
            :class:`pandas.DataFrame` containing the data

        """
        col_names = ['year', 'month', 'day', 'hour', 'minute',
                     'wind_direction', 'wind_speed', 'wind_gust',
                     'wave_height', 'dominant_wave_period', 'average_wave_period',
                     'dominant_wave_direction', 'pressure',
                     'air_temperature', 'water_temperature', 'dewpoint',
                     'visibility', '3hr_pressure_tendency', 'water_level_above_mean']

        col_units = {'wind_direction': 'degrees',
                     'wind_speed': 'meters/second',
                     'wind_gust': 'meters/second',
                     'wave_height': 'meters',
                     'dominant_wave_period': 'seconds',
                     'average_wave_period': 'seconds',
                     'dominant_wave_direction': 'degrees',
                     'pressure': 'hPa',
                     'air_temperature': 'degC',
                     'water_temperature': 'degC',
                     'dewpoint': 'degC',
                     'visibility': 'nautical_mile',
                     '3hr_pressure_tendency': 'hPa',
                     'water_level_above_mean': 'feet',
                     'time': None}

        df = pd.read_table(StringIO(content), comment='#', na_values='MM',
                           names=col_names, sep=r'\s+')
        df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']], utc=True)
        df = df.drop(columns=['year', 'month', 'day', 'hour', 'minute'])
        df.units = col_units
        return df