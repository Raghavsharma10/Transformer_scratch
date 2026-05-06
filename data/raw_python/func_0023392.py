def from_times(cls, times, delta_t=DEFAULT_OBSERVATION_TIME):
        """
        Create a TimeMOC from a `astropy.time.Time`

        Parameters
        ----------
        times : `astropy.time.Time`
            astropy observation times
        delta_t : `astropy.time.TimeDelta`, optional
            the duration of one observation. It is set to 30 min by default. This data is used to compute the
            more efficient TimeMOC order to represent the observations (Best order = the less precise order which
            is able to discriminate two observations separated by ``delta_t``).

        Returns
        -------
        time_moc : `~mocpy.tmoc.TimeMOC`
        """
        times_arr = np.asarray(times.jd * TimeMOC.DAY_MICRO_SEC, dtype=int)
        intervals_arr = np.vstack((times_arr, times_arr + 1)).T

        # degrade the TimeMoc to the order computer from ``delta_t``
        order = TimeMOC.time_resolution_to_order(delta_t)
        return TimeMOC(IntervalSet(intervals_arr)).degrade_to_order(order)