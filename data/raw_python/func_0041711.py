def duration_so_far(self):
        """Return how the duration so far.

        :returns: the duration from the time the Interval was started if the
            interval is running, otherwise ``False``.
        """
        if self._start_instant is None:
            return False
        if self._stop_instant is None:
            return int((instant() - self._start_instant) * 1000)
        return False