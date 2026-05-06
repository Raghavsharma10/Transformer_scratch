def append_sources_from(self, other):
        """Merge the source alias lists of two CatDicts."""
        # Get aliases lists from this `CatDict` and other
        self_aliases = self[self._KEYS.SOURCE].split(',')
        other_aliases = other[self._KEYS.SOURCE].split(',')

        # Store alias to `self`
        self[self._KEYS.SOURCE] = uniq_cdl(self_aliases + other_aliases)

        return