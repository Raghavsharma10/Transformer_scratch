def iteritems(self, pattern="*"):
        """An iterator over all the ``(key, value)`` items in the database,
        or where the keys matches ``pattern``."""
        for key in self.keys(pattern):
            yield (key, self[key])