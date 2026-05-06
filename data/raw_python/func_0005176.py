def add_interval(self, precision=0):
        """ Adds an interval to :prop:intervals
            -> #str formatted time
        """
        precision = precision or self.precision
        interval = round((self._stop - self._start), precision)
        self.intervals.append(interval)
        self._intervals_len += 1
        self._start = time.perf_counter()
        return self.format_time(interval)