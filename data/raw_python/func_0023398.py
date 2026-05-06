def total_duration(self):
        """
        Get the total duration covered by the temporal moc

        Returns
        -------
        duration : `~astropy.time.TimeDelta`
            total duration of all the observation times of the tmoc
            total duration of all the observation times of the tmoc

        """

        if self._interval_set.empty():
            return 0

        total_time_us = 0
        # The interval set is checked for consistency before looping over all the intervals
        for (start_time, stop_time) in self._interval_set._intervals:
            total_time_us = total_time_us + (stop_time - start_time)

        duration = TimeDelta(total_time_us / 1e6, format='sec', scale='tdb')
        return duration