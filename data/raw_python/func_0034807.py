def _make_resource(self):
        """
        Returns a resource instance.
        """
        with self._lock:
            for i in self._unavailable_range():
                if self._reference_queue[i] is None:
                    rtracker = _ResourceTracker(
                        self._factory(**self._factory_arguments))

                    self._reference_queue[i] = rtracker
                    self._size += 1

                    return rtracker

            raise PoolFullError