def clear(self):
        """Remove all key-value pairs."""
        for i in range(self.maxlevel):
            self._head[2+i] = self._tail
            self._tail[-1] = 0
        self._level = 1