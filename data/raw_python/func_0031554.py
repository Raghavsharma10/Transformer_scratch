def difference(self, other):
        """
        Subtract an Interval or IntervalSet from the intervals in the set.
        """
        intervals = other if isinstance(other, IntervalSet) else IntervalSet((other,))
        result = IntervalSet()
        for left in self:
            for right in intervals:
                left = left - right
            if isinstance(left, IntervalSet):
                for interval in left:
                    result.add(interval)
            else:
                result.add(left)
        return result