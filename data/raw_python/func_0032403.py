def simulate(self):
        """Generates a random integer in the available range."""
        min_ = (-sys.maxsize - 1) if self._min is None else self._min
        max_ = sys.maxsize if self._max is None else self._max
        return random.randint(min_, max_)