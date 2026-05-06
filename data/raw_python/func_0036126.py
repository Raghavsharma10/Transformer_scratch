def range_by_score(self, min, max):
        """Return all the elements with score >= min and score <= max
        (a range query) from the sorted set."""
        data = self.items()
        keys = [r[1] for r in data] 
        start = bisect.bisect_left(keys, min)
        end = bisect.bisect_right(keys, max, start)
        return self._as_set()[start:end]