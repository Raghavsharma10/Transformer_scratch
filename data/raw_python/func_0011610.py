def count(self):
        """Return a count of instances."""
        if self._primary_keys is None:
            return self.queryset.count()
        else:
            return len(self.pks)