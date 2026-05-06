def set(self, value):
        """Set the value of the bar. If the value is out of bound, sets it to an extremum"""
        value = min(self.max, max(self.min, value))
        self._value = value
        start_new_thread(self.func, (self.get(),))