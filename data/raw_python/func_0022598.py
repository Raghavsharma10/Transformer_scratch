def gauge(self, slug, current_value):
        """Set the value for a Gauge.

        * ``slug`` -- the unique identifier (or key) for the Gauge
        * ``current_value`` -- the value that the gauge should display

        """
        k = self._gauge_key(slug)
        self.r.sadd(self._gauge_slugs_key, slug)  # keep track of all Gauges
        self.r.set(k, current_value)