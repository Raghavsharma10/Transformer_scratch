def start_fsweep(self, start=None, stop=None, step=None):
        """Starts a frequency sweep.

        :param start: Sets the start frequency.
        :param stop: Sets the target frequency.
        :param step: Sets the frequency step.

        """
        if start:
            self.frequency_start = start
        if stop:
            self.frequency_stop = stop
        if step:
            self.frequency_step = step
        self._write(('SWEEP', Integer), 1)