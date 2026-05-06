def overlaps(self, other):
        """If self and other have any overlapping values returns True, otherwise returns False"""
        if self > other:
            smaller, larger = other, self
        else:
            smaller, larger = self, other
        if larger.empty():
            return False
        if smaller._upper_value == larger._lower_value:
            return smaller._upper == smaller.CLOSED and larger._lower == smaller.CLOSED
        return larger._lower_value < smaller._upper_value