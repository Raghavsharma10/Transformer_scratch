def intersection(self, other):
        """Returns a new IntervalSet which represents the intersection of each of the intervals in this IntervalSet
        with each of the intervals in the other IntervalSet.
        :param other: An IntervalSet to intersect with this one.
        """
        # if self or other is empty the intersection will be empty
        result = IntervalSet()
        for other_inter in other:
            for interval in self:
                this_intervals_intersection = other_inter.intersect(interval)
                if this_intervals_intersection is not None:
                    result._add(this_intervals_intersection)
        return result