def _sorted_keys(self):
        """
        Return list of keys sorted by version

        Sorting is done based on :py:func:`pkg_resources.parse_version`
        """

        try:
            keys = self._cache['sorted_keys']
        except KeyError:
            keys = self._cache['sorted_keys'] = sorted(self.keys(), key=parse_version)

        return keys