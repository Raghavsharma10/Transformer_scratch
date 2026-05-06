def request_all_data(cls, time, pressure=None, **kwargs):
        """Retrieve upper air observations from Iowa State's archive for all stations.

        Parameters
        ----------
        time : datetime
            The date and time of the desired observation.

        pressure : float, optional
            The mandatory pressure level at which to request data (in hPa). If none is given,
            all the available data in the profiles is returned.

        kwargs
            Arbitrary keyword arguments to use to initialize source

        Returns
        -------
            :class:`pandas.DataFrame` containing the data

        """
        endpoint = cls()
        df = endpoint._get_data(time, None, pressure, **kwargs)
        return df