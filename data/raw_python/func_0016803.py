def reset(self):
        """Reset to the initial state, clearing the buffer and zeroing count and scanned."""
        self.buffer.clear()
        self._count = 0
        self._scanned = 0
        self._exhausted = False
        self.request.pop("ExclusiveStartKey", None)