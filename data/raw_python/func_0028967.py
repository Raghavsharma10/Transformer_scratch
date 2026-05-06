def do_counter_conversion(self):
        """Update latest value to the diff between it and the previous value"""
        if self.is_counter:
            if self._previous_counter_value is None:
                prev_value = self.latest_value
            else:
                prev_value = self._previous_counter_value
            self._previous_counter_value = self.latest_value
            self.latest_value = self.latest_value - prev_value