def value_eq(self, other):
        """Sorted comparison of values."""
        self_sorted = ordered.ordered(self.getvalues())
        other_sorted = ordered.ordered(repeated.getvalues(other))
        return self_sorted == other_sorted