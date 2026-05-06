def get_version(self, extra=None):
        """
        This will return a string that can be used as a prefix
        for django's cache key. Something like key.1 or key.1.2

        If a version was not found '1' will be stored and returned as
        the number for that key.

        If extra is given a version will be returned for that value.
        Otherwise the major version will be returned.

        :param extra: the minor version to get. Defaults to None.
        """

        if extra:
            key = self._get_extra_key(extra)
        else:
            key = self.key

        v = self._get_cache(key).get(key)
        if v == None:
            v = self._increment_version(extra=extra)

        return "%s.%s" % (key, v)