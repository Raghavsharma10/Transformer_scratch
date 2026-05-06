def merge(self, other):
        """Merge another translation into this catalog."""
        if not getattr(other, '_catalog', None):
            return  # NullTranslations() has no _catalog
        if self._catalog is None:
            # Take plural and _info from first catalog found
            self.plural = other.plural
            self._info = other._info.copy()
            self._catalog = other._catalog.copy()
        else:
            self._catalog.update(other._catalog)