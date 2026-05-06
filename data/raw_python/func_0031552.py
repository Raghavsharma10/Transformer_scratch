def union(self, other):
        """Returns a new IntervalSet which represents the union of each of the intervals in this IntervalSet with each
        of the intervals in the other IntervalSet
        :param other: An IntervalSet to union with this one.
        """
        result = IntervalSet()
        for el in self:
            result.add(el)
        for el in other:
            result.add(el)
        return result