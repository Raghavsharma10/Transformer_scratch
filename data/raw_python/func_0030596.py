def report_progress(self):
        """
        This function can be called from a higher level to report progress. It is usually called from an alarm
        signal handler which is installed just before starting an operation:

        :return: Tuple: (process description, #records, #total records, #rate)
        """
        from time import time

        # rows, rate = pl.sink.report_progress()
        return (self.i, round(float(self.i) / float(time() - self._start_time), 2))