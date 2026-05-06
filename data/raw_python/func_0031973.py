def difference(self, other):
        """Returns a new Interval or an :class:`~pyinter.IntervalSet` representing the subtraction of this
        :class:`~pyinter.Interval` with the other :class:`~pyinter.Interval`.

        The result will contain everything that is contained by the left interval but not contained
        by the second interval.

        If the `other` interval is enclosed in this one then this will return a
        :class:`~pyinter.IntervalSet`, otherwise this returns a :class:`~pyinter.Interval`.
        """
        if other.empty():
            return self
        if self in other:
            return open(self._lower_value, self._lower_value)
        if self._lower == other._lower and self._lower_value == other._lower_value:
            return Interval(self._opposite_boundary_type(other._upper), other._upper_value, self._upper_value, self._upper)
        if self._upper == other._upper and self._upper_value == other._upper_value:
            return Interval(self._lower, self._lower_value, other._lower_value, self._opposite_boundary_type(other._lower))
        if other in self:
            return IntervalSet([
                Interval(self._lower, self._lower_value, other.lower_value, self._opposite_boundary_type(other._lower)),
                Interval(self._opposite_boundary_type(other._upper), other._upper_value, self.upper_value, self._upper),
            ])
        if other.lower_value in self:
            return Interval(self._lower, self._lower_value, other._lower_value, self._opposite_boundary_type(other._lower))
        if other.upper_value in self:
            return Interval(self._opposite_boundary_type(other._upper), other._upper_value, self._upper_value, self._upper)
        return Interval(self._lower, self._lower_value, self._upper_value, self._upper)