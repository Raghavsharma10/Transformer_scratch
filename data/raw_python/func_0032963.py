def range(self, value):
        """Estimates an appropriate sensitivity range."""
        self._buffer.append(abs(value))
        mean = sum(self._buffer) / len(self._buffer)
        estimate = next(
            (r for r in self.ranges if mean < self.scale * r),
            self.ranges[-1]
        )
        if self._mapping:
            return self._mapping[estimate]
        else:
            return estimate