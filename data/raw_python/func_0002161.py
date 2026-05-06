def raw_buoy_data(cls, buoy, data_type='txt'):
        """Retrieve the raw buoy data contents from NDBC.

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
            'data_spec' raw spectral wave data
            'swdir' spectral wave data (alpha1)
            'swdir2' spectral wave data (alpha2)
            'swr1' spectral wave data (r1)
            'swr2' spectral wave data (r2)
            'adcp' acoustic doppler current profiler
            'ocean' oceanographic data
            'tide' tide data
            'srad' solar radiation data
            'dart' water column height
            'supl' supplemental measurements data
            'rain' hourly rain data

        Returns
        -------
        Raw data string

        """
        endpoint = cls()
        resp = endpoint.get_path('data/realtime2/{}.{}'.format(buoy, data_type))
        return resp.text