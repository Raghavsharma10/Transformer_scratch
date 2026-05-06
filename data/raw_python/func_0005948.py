def get(self, key, default=None, as_int=False, setter=None):
        """Gets a value from the cache.

        :param str|unicode key: The cache key to get value for.

        :param default: Value to return if none found in cache.

        :param bool as_int: Return 64bit number instead of str.

        :param callable setter: Setter callable to automatically set cache
            value if not already cached. Required to accept a key and return
            a value that will be cached.

        :rtype: str|unicode|int

        """
        if as_int:
            val = uwsgi.cache_num(key, self.name)
        else:
            val = decode(uwsgi.cache_get(key, self.name))

        if val is None:

            if setter is None:
                return default

            val = setter(key)

            if val is None:
                return default

            self.set(key, val)

        return val