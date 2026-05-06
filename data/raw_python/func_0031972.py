def intersect(self, other):
        """Returns a new :class:`~pyinter.Interval` representing the intersection of this :class:`~pyinter.Interval`
        with the other :class:`~pyinter.Interval`"""
        if self.overlaps(other):
            newlower_value = max(self.lower_value, other.lower_value)
            new_upper_value = min(self._upper_value, other._upper_value)
            new_lower, new_upper = self._get_new_lower_upper(other, self.intersect)
            return Interval(new_lower, newlower_value, new_upper_value, new_upper)
        else:
            return None