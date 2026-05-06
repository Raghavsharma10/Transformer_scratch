def add_file(self, filepath, gzip=False, cache_name=None):
        """Load a static file in the cache.

        .. note:: Items are stored with the filepath as is (relative or absolute) as the key.

        :param str|unicode filepath:

        :param bool gzip: Use gzip compression.

        :param str|unicode cache_name: If not set, default will be used.

        """
        command = 'load-file-in-cache'

        if gzip:
            command += '-gzip'

        cache_name = cache_name or ''
        value = '%s %s' % (cache_name, filepath)

        self._set(command, value.strip(), multi=True)

        return self._section