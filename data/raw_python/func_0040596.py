def intersection(self,other):
        """
        Return a new IntervalSet with the intersection of the two sets, i.e.
        all elements that are both in self and other.

        :param IntervalSet other: Set to intersect with
        :rtype: IntervalSet
        """
        res = []
        for i1 in self.ints:
            for i2 in other.ints:
                res.append(i1.intersection(i2))

        return IntervalSet(res)