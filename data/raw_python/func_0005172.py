def start(self):
        """ Starts the timer """
        if not self._start:
            self._first_start = time.perf_counter()
            self._start = self._first_start
        else:
            self._start = time.perf_counter()