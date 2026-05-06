def _get_data(self, time, site_id, pressure=None):
        """Download data from Iowa State's upper air archive.

        Parameters
        ----------
        time : datetime
            Date and time for which data should be downloaded
        site_id : str
            Site id for which data should be downloaded
        pressure : float, optional
            Mandatory pressure level at which to request data (in hPa).

        Returns
        -------
            :class:`pandas.DataFrame` containing the data

        """
        json_data = self._get_data_raw(time, site_id, pressure)
        data = {}
        for profile in json_data['profiles']:
            for pt in profile['profile']:
                for field in ('drct', 'dwpc', 'hght', 'pres', 'sknt', 'tmpc'):
                    data.setdefault(field, []).append(np.nan if pt[field] is None
                                                      else pt[field])
                for field in ('station', 'valid'):
                    data.setdefault(field, []).append(np.nan if profile[field] is None
                                                      else profile[field])

        # Make sure that the first entry has a valid temperature and dewpoint
        idx = np.argmax(~(np.isnan(data['tmpc']) | np.isnan(data['dwpc'])))

        # Stuff data into a pandas dataframe
        df = pd.DataFrame()
        df['pressure'] = ma.masked_invalid(data['pres'][idx:])
        df['height'] = ma.masked_invalid(data['hght'][idx:])
        df['temperature'] = ma.masked_invalid(data['tmpc'][idx:])
        df['dewpoint'] = ma.masked_invalid(data['dwpc'][idx:])
        df['direction'] = ma.masked_invalid(data['drct'][idx:])
        df['speed'] = ma.masked_invalid(data['sknt'][idx:])
        df['station'] = data['station'][idx:]
        df['time'] = [datetime.strptime(valid, '%Y-%m-%dT%H:%M:%SZ')
                      for valid in data['valid'][idx:]]

        # Calculate the u and v winds
        df['u_wind'], df['v_wind'] = get_wind_components(df['speed'],
                                                         np.deg2rad(df['direction']))

        # Drop any rows with all NaN values for T, Td, winds
        df = df.dropna(subset=('temperature', 'dewpoint', 'direction', 'speed',
                               'u_wind', 'v_wind'), how='all').reset_index(drop=True)

        # Add unit dictionary
        df.units = {'pressure': 'hPa',
                    'height': 'meter',
                    'temperature': 'degC',
                    'dewpoint': 'degC',
                    'direction': 'degrees',
                    'speed': 'knot',
                    'u_wind': 'knot',
                    'v_wind': 'knot',
                    'station': None,
                    'time': None}
        return df