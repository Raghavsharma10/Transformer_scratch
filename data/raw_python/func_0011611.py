def filter(self, **kwargs):
        """Filter the base queryset."""
        assert not self._primary_keys
        self.queryset = self.queryset.filter(**kwargs)
        return self