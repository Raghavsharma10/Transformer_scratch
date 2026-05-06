def step_amp(self):
        """
        Change the amplitude according to the change rate and drift target.

        Returns: None
        """
        difference = self.drift_target - self._raw_value
        if abs(difference) < self.change_rate:
            self.value = self.drift_target
        else:
            delta = self.change_rate * numpy.sign(difference)
            self.value = self._raw_value + delta