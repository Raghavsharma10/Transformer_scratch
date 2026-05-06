def _add(self, other):
        """Add a interval to the underlying IntervalSet data store. This does not perform any tests as we assume that
        any requirements have already been checked and that this function is being called by an internal function such
        as union(), intersection() or add().
        :param other: An Interval to add to this one
        """
        if len([interval for interval in self if other in interval]) > 0:  # if other is already represented
            return
        # remove any intervals which are fully represented by the interval we are adding
        to_remove = [interval for interval in self if interval in other]
        self._data.difference_update(to_remove)
        self._data.add(other)