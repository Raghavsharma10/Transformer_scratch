def difference(self,other):
        """
        Return a new IntervalSet with the difference of the two sets, i.e.
        all elements that are in self but not in other.

        :param IntervalSet other: Set to subtract
        :rtype: IntervalSet
        """
        res = IntervalSet.everything()
        for j in other.ints:
            tmp = []
            for i in self.ints:
                tmp.extend(i._difference(j))
            res = res.intersection(IntervalSet(tmp))
        return res