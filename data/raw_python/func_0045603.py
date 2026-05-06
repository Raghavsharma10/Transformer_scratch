def clear(self):
        """Clears state so it can be used for generating entirely new
        instructions."""
        if not self._clear:
            self.lib._jit_clear_state(self.state)
            self._clear = True