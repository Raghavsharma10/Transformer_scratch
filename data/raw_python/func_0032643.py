def start_asweep(self, start=None, stop=None, step=None):
        """Starts a amplitude sweep.

        :param start: Sets the start frequency.
        :param stop: Sets the target frequency.
        :param step: Sets the frequency step.

        """
        if start:
            self.amplitude_start = start
        if stop:
            self.amplitude_stop = stop
        if step:
            self.amplitude_step = step
        self._write(('SWEEP', Integer), 2)