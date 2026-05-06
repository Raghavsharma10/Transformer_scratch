def trig_count(self, value):
        """Set the numbers of BCIDs (usually 16) of one event."""
        self._trig_count = 16 if value == 0 else value
        self.interpreter.set_trig_count(self._trig_count)