def merge(self, other):
        """Recursively merge tags from another compound."""
        for key, value in other.items():
            if key in self and (isinstance(self[key], Compound)
                                and isinstance(value, dict)):
                self[key].merge(value)
            else:
                self[key] = value