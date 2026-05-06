def generate_bangbang_output(self):
        """Generates the latest on or off pulse in
           the string of on (True) or off (False) pulses
           according to the desired heat_level setting.  Successive calls
           to this function will return the next value in the
           on/off array series.  Call this at control loop rate to
           obtain the necessary on/off pulse train.
           This system will not work if the caller expects to be able
           to specify a new heat_level at every control loop iteration.
           Only the value set at every number_of_segments iterations
           will be picked up for output! Call about_to_rollover to determine
           if it's time to set a new heat_level, if a new level is desired."""
        if self._current_index >= self._num_segments:
            # we're due to switch over to the next
            # commanded heat_level
            self._heat_level_now = self._heat_level
            # reset array index
            self._current_index = 0
        # return output
        out = self._output_array[self._heat_level_now][self._current_index]
        self._current_index += 1
        return out