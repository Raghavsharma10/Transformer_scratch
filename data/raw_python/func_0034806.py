def _harvest_lost_resources(self):
        """Return lost resources to pool."""
        with self._lock:
            for i in self._unavailable_range():
                rtracker = self._reference_queue[i]
                if rtracker is not None and rtracker.available():
                    self.put_resource(rtracker.resource)