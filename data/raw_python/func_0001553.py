def issuperset(self, other):
        """Report whether this RangeSet contains another set."""
        self._binary_sanity_check(other)
        return set.issuperset(self, other)