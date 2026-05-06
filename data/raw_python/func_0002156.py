def realtime_observations(cls, buoy, data_type='txt'):
        """Retrieve the realtime buoy data from NDBC.

        Parameters
        ----------
        buoy : str
            Name of buoy
        data_type : str
            Type of data requested, must be one of
            'txt' standard meteorological data
            'drift' meteorological data from drifting buoys and limited moored buoy data
            mainly from international partners
            'cwind' continuous winds data (10 minute average)
            'spec' spectral wave summaries
            'ocean' oceanographic data
            'srad' solar radiation data
            'dart' water column height
            'supl' supplemental measurements data
            'rain' hourly rain data

        Returns
        -------
            Raw data string

        """
        endpoint = cls()
        parsers = {'txt': endpoint._parse_met,
                   'drift': endpoint._parse_drift,
                   'cwind': endpoint._parse_cwind,
                   'spec': endpoint._parse_spec,
                   'ocean': endpoint._parse_ocean,
                   'srad': endpoint._parse_srad,
                   'dart': endpoint._parse_dart,
                   'supl': endpoint._parse_supl,
                   'rain': endpoint._parse_rain}

        if data_type not in parsers:
            raise KeyError('Data type must be txt, drift, cwind, spec, ocean, srad, dart,'
                           'supl, or rain for parsed realtime data.')

        raw_data = endpoint.raw_buoy_data(buoy, data_type=data_type)
        return parsers[data_type](raw_data)