def serialize_numeric(self, tag):
        """Return the literal representation of a numeric tag."""
        str_func = int.__str__ if isinstance(tag, int) else float.__str__
        return str_func(tag) + tag.suffix