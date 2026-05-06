def rate(self):
        """Report the insertion rate in records per second"""

        end = self._end_time if self._end_time else time.time()

        return self._count / (end - self._start_time)