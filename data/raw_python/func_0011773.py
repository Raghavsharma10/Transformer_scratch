def heat_level(self, value):
        """Set the desired output level. Must be between 0 and
        number_of_segments inclusive."""
        if value < 0:
            self._heat_level = 0
        elif round(value) > self._num_segments:
            self._heat_level = self._num_segments
        else:
            self._heat_level = int(round(value))