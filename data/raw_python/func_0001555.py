def update(self, iterable):
        """Add all integers from an iterable (such as a list)."""
        if isinstance(iterable, RangeSet):
            # keep padding unless is has not been defined yet
            if self.padding is None and iterable.padding is not None:
                self.padding = iterable.padding
        assert type(iterable) is not str
        set.update(self, iterable)