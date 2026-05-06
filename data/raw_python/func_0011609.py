def pks(self):
        """Lazy-load the primary keys."""
        if self._primary_keys is None:
            self._primary_keys = list(
                self.queryset.values_list('pk', flat=True))
        return self._primary_keys