def difference(self, other):
        """Return the difference of two RangeSets as a new RangeSet.

        (I.e. all elements that are in this set and not in the other.)
        """
        #NOTE: This is a work around
        # Python 3 return as the result of set.intersection a new set instance.
        # Python 2 however returns as a the result a ClusterShell.RangeSet.RangeSet instance.
        # ORIGINAL CODE: return self._wrap_set_op(set.difference, other)
        copy = self.copy()
        copy.difference_update(other)
        return copy