def _put(self, timestamp, value):
        """Replace the value associated with "timestamp" or add the new value"""

        idx = self._lookup(timestamp)
        if idx is not None:
            self._values[idx] = (timestamp, value)
        else:
            self._values.append((timestamp, value))