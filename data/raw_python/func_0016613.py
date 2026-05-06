def is_valid(self, name, value):
        """
        Returns true if the given value matches the type for the given name
        according to the schema.
        Returns false otherwise.
        """
        if name not in self.header_types:
            return False
        t = self.header_types[name]
        if t == ATTR_TYPE_DISCRETE:
            return isinstance(value, int)
        elif t == ATTR_TYPE_CONTINUOUS:
            return isinstance(value, (float, Decimal))
        return True