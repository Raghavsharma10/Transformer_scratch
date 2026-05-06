def max_time(self):
        """
        Get the `~astropy.time.Time` time of the tmoc last observation

        Returns
        -------
        max_time : `~astropy.time.Time`
            time of the last observation

        """

        max_time = Time(self._interval_set.max / TimeMOC.DAY_MICRO_SEC, format='jd', scale='tdb')
        return max_time