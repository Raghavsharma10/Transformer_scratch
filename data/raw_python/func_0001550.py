def intersection(self, other):
        """Return the intersection of two RangeSets as a new RangeSet.

        (I.e. all elements that are in both sets.)
        """
        #NOTE: This is a work around 
        # Python 3 return as the result of set.intersection a new set instance.
        # Python 2 however returns as a the result a ClusterShell.RangeSet.RangeSet instance.
        # ORIGINAL CODE: return self._wrap_set_op(set.intersection, other)
        copy = self.copy()
        copy.intersection_update(other)
        return copy