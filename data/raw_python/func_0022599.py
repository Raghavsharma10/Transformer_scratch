def delete_gauge(self, slug):
        """Removes all gauges with the given ``slug``."""
        key = self._gauge_key(slug)
        self.r.delete(key)  # Remove the Gauge
        self.r.srem(self._gauge_slugs_key, slug)