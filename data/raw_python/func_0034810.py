def _unavailable_range(self):
        """
        Return a generator for the indices of the unavailable region of
        ``_reference_queue``.
        """
        with self._lock:
            i = self._resource_end
            j = self._resource_start
            if j < i or self.empty():
                j += self.maxsize

            for k in range(i, j):
                yield k % self.maxsize