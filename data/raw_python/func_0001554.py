def difference_update(self, other, strict=False):
        """Remove all elements of another set from this RangeSet.
        
        If strict is True, raise KeyError if an element cannot be removed.
        (strict is a RangeSet addition)"""
        if strict and other not in self:
            raise KeyError(other.difference(self)[0])
        set.difference_update(self, other)