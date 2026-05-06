def min_time(self):
        """
        Get the `~astropy.time.Time` time of the tmoc first observation

        Returns
        -------
        min_time : `astropy.time.Time`
            time of the first observation

        """

        min_time = Time(self._interval_set.min / TimeMOC.DAY_MICRO_SEC, format='jd', scale='tdb')
        return min_time