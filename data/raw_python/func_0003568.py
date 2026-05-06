def state(self, new_state):
        """Set the state."""
        with self.lock:
            self._state.exit()
            self._state = new_state
            self._state.enter()