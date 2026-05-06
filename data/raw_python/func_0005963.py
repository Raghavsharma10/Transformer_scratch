def add_item(self, key, value, cache_name=None):
        """Add an item into the given cache.

        This is a commodity option (mainly useful for testing) allowing you
        to store an item in a uWSGI cache during startup.

        :param str|unicode key:

        :param value:

        :param str|unicode cache_name: If not set, default will be used.

        """
        cache_name = cache_name or ''
        value = '%s %s=%s' % (cache_name, key, value)

        self._set('add-cache-item', value.strip(), multi=True)

        return self._section