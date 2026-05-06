def issubset(self, other):
        """Report whether another set contains this RangeSet."""
        self._binary_sanity_check(other)
        return set.issubset(self, other)