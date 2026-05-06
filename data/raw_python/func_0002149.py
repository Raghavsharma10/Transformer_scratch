def request_data(cls, time, site_id, derived=False):
        """Retreive IGRA version 2 data for one station.

        Parameters
        --------
        site_id : str
            11-character IGRA2 station identifier.

        time : datetime
           The date and time of the desired observation. If list of two times is given,
           dataframes for all dates within the two dates will be returned.

        Returns
        -------
            :class: `pandas.DataFrame` containing the data.

        """
        igra2 = cls()

        # Set parameters for data query
        if derived:
            igra2.ftpsite = igra2.ftpsite + 'derived/derived-por/'
            igra2.suffix = igra2.suffix + '-drvd.txt'
        else:
            igra2.ftpsite = igra2.ftpsite + 'data/data-por/'
            igra2.suffix = igra2.suffix + '-data.txt'

        if type(time) == datetime.datetime:
            igra2.begin_date = time
            igra2.end_date = time
        else:
            igra2.begin_date, igra2.end_date = time

        igra2.site_id = site_id

        df, headers = igra2._get_data()

        return df, headers