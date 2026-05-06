def convert_items(self, items):
        """Generator like `convert_iterable`, but for 2-tuple iterators."""
        return ((key, self.convert(value, self)) for key, value in items)