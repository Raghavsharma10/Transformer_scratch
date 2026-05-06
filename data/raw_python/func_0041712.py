def duration(self):
        """Returns the integer value of the interval, the value is in milliseconds.

        If the interval has not had stop called yet,
        it will report the number of milliseconds in the interval up to the current point in time.
        """
        if self._stop_instant is None:
            return int((instant() - self._start_instant) * 1000)
        if self._duration is None:
            self._duration = int((self._stop_instant - self._start_instant) * 1000)
        return self._duration