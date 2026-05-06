def add(self, other):
        """
        Add an Interval to the IntervalSet by taking the union of the given Interval object with the existing
        Interval objects in self.

        This has no effect if the Interval is already represented.
        :param other: an Interval to add to this IntervalSet.
        """
        if other.empty():
            return

        to_add = set()
        for inter in self:
            if inter.overlaps(other):  # if it overlaps with this interval then the union will be a single interval
                to_add.add(inter.union(other))
        if len(to_add) == 0:  # other must not overlap with any interval in self (self could be empty!)
            to_add.add(other)
        # Now add the intervals found to self
        if len(to_add) > 1:
            set_to_add = IntervalSet(to_add)  # creating an interval set unions any overlapping intervals
            for el in set_to_add:
                self._add(el)
        elif len(to_add) == 1:
            self._add(to_add.pop())