def add(self, labels, value):
        """Add adds a single observation to the summary."""

        if type(value) not in (float, int):
            raise TypeError("Summary only works with digits (int, float)")

        # We have already a lock for data but not for the estimator
        with mutex:
            try:
                e = self.get_value(labels)
            except KeyError:
                # Initialize quantile estimator
                e = quantile.Estimator(*self.__class__.DEFAULT_INVARIANTS)
                self.set_value(labels, e)
            e.observe(float(value))