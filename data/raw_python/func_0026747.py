def get_stub(self):
        """Get a new `Entry` which contains the 'stub' of this one.

        The 'stub' is only the name and aliases.

        Usage:
        -----
        To convert a normal entry into a stub (for example), overwrite the
        entry in place, i.e.
        >>> entries[name] = entries[name].get_stub()

        Returns
        -------
        stub : `astrocats.catalog.entry.Entry` subclass object
            The type of the returned object is this instance's type.

        """
        stub = type(self)(self.catalog, self[self._KEYS.NAME], stub=True)
        if self._KEYS.ALIAS in self:
            stub[self._KEYS.ALIAS] = self[self._KEYS.ALIAS]
        if self._KEYS.DISTINCT_FROM in self:
            stub[self._KEYS.DISTINCT_FROM] = self[self._KEYS.DISTINCT_FROM]
        if self._KEYS.RA in self:
            stub[self._KEYS.RA] = self[self._KEYS.RA]
        if self._KEYS.DEC in self:
            stub[self._KEYS.DEC] = self[self._KEYS.DEC]
        if self._KEYS.DISCOVER_DATE in self:
            stub[self._KEYS.DISCOVER_DATE] = self[self._KEYS.DISCOVER_DATE]
        if self._KEYS.SOURCES in self:
            stub[self._KEYS.SOURCES] = self[self._KEYS.SOURCES]
        return stub