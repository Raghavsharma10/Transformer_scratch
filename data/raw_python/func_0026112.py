def _getCacheEntry(self, name):
        """
        :param name: the name of the cache entry.
        :return: the entry or none.
        """
        return next((x for x in self._uploadCache if x['name'] == name), None)