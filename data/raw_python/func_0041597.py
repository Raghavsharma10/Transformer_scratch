def equality(self, other):
        """Calculate equality based on equality of all group items."""
        if not len(self) == len(other):
            return False
        return super().equality(other)