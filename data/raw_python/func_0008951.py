def clear(self):
        """Resets the object at its initial (empty) state."""
        self._deque.clear()
        self._total_length = 0
        self._has_view = False