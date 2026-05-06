def load(self, *args, **kwargs):
        """
        Load instrument data into instrument object.data

        (Wraps pysat.Instrument.load; documentation of that function is
        reproduced here.)

        Parameters
        ---------
        yr : integer
            Year for desired data
        doy : integer
            day of year
        data : datetime object
            date to load
        fname : 'string'
            filename to be loaded
        verifyPad : boolean
            if true, padding data not removed (debug purposes)
        """

        for instrument in self.instruments:
            instrument.load(*args, **kwargs)