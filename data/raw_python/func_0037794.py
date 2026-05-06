def apply(self, collection, ops, **kwargs):
        """Apply the filter to collection."""
        validator = lambda obj: all(op(obj, val) for (op, val) in ops)  # noqa
        return [o for o in collection if validator(o)]