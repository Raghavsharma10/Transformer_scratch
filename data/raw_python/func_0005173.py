def stop(self, precision=0):
        """ Stops the timer, adds it as an interval to :prop:intervals
            @precision: #int number of decimal places to round to

            -> #str formatted interval time
        """
        self._stop = time.perf_counter()
        return self.add_interval(precision)