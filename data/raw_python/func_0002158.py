def _parse_supl(content):
        """Parse supplemental measurements data.

        Parameters
        ----------
        content : str
            Data to parse

        Returns
        -------
            :class:`pandas.DataFrame` containing the data

        """
        col_names = ['year', 'month', 'day', 'hour', 'minute',
                     'hourly_low_pressure', 'hourly_low_pressure_time',
                     'hourly_high_wind', 'hourly_high_wind_direction',
                     'hourly_high_wind_time']

        col_units = {'hourly_low_pressure': 'hPa',
                     'hourly_low_pressure_time': None,
                     'hourly_high_wind': 'meters/second',
                     'hourly_high_wind_direction': 'degrees',
                     'hourly_high_wind_time': None,
                     'time': None}

        df = pd.read_table(StringIO(content), comment='#', na_values='MM',
                           names=col_names, sep=r'\s+')

        df['time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']], utc=True)

        df['hours'] = np.floor(df['hourly_low_pressure_time'] / 100)
        df['minutes'] = df['hourly_low_pressure_time'] - df['hours'] * 100
        df['hours'] = df['hours'].replace(99, np.nan)
        df['minutes'] = df['minutes'].replace(99, np.nan)
        df['hourly_low_pressure_time'] = pd.to_datetime(df[['year', 'month', 'day', 'hours',
                                                            'minutes']], utc=True)

        df['hours'] = np.floor(df['hourly_high_wind_time'] / 100)
        df['minutes'] = df['hourly_high_wind_time'] - df['hours'] * 100
        df['hours'] = df['hours'].replace(99, np.nan)
        df['minutes'] = df['minutes'].replace(99, np.nan)
        df['hourly_high_wind_time'] = pd.to_datetime(df[['year', 'month', 'day',
                                                         'hours', 'minutes']], utc=True)
        df = df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'hours', 'minutes'])
        df.units = col_units
        return df