def buoy_data_types(cls, buoy):
        """Determine which types of data are available for a given buoy.

        Parameters
        ----------
        buoy : str
            Buoy name

        Returns
        -------
            dict of valid file extensions and their descriptions

        """
        endpoint = cls()
        file_types = {'txt': 'standard meteorological data',
                      'drift': 'meteorological data from drifting buoys and limited moored'
                               'buoy data mainly from international partners',
                      'cwind': 'continuous wind data (10 minute average)',
                      'spec': 'spectral wave summaries',
                      'data_spec': 'raw spectral wave data',
                      'swdir': 'spectral wave data (alpha1)',
                      'swdir2': 'spectral wave data (alpha2)',
                      'swr1': 'spectral wave data (r1)',
                      'swr2': 'spectral wave data (r2)',
                      'adcp': 'acoustic doppler current profiler',
                      'ocean': 'oceanographic data',
                      'tide': 'tide data',
                      'srad': 'solar radiation data',
                      'dart': 'water column height',
                      'supl': 'supplemental measurements data',
                      'rain': 'hourly rain data'}
        available_data = {}
        buoy_url = 'https://www.ndbc.noaa.gov/data/realtime2/' + buoy + '.'
        for key in file_types:
            if endpoint._check_if_url_valid(buoy_url + key):
                available_data[key] = file_types[key]
        return available_data