def from_time_ranges(cls, min_times, max_times, delta_t=DEFAULT_OBSERVATION_TIME):
        """
        Create a TimeMOC from a range defined by two `astropy.time.Time`

        Parameters
        ----------
        min_times : `astropy.time.Time`
            astropy times defining the left part of the intervals
        max_times : `astropy.time.Time`
            astropy times defining the right part of the intervals
        delta_t : `astropy.time.TimeDelta`, optional
            the duration of one observation. It is set to 30 min by default. This data is used to compute the
            more efficient TimeMOC order to represent the observations (Best order = the less precise order which
            is able to discriminate two observations separated by ``delta_t``).

        Returns
        -------
        time_moc : `~mocpy.tmoc.TimeMOC`
        """
        min_times_arr = np.asarray(min_times.jd * TimeMOC.DAY_MICRO_SEC, dtype=int)
        max_times_arr = np.asarray(max_times.jd * TimeMOC.DAY_MICRO_SEC, dtype=int)

        intervals_arr = np.vstack((min_times_arr, max_times_arr + 1)).T

        # degrade the TimeMoc to the order computer from ``delta_t``
        order = TimeMOC.time_resolution_to_order(delta_t)
        return TimeMOC(IntervalSet(intervals_arr)).degrade_to_order(order)