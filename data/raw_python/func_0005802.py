def set_paths_caching_params(self, timeout=None, cache_name=None):
        """Use the uWSGI caching subsystem to store mappings from URI to filesystem paths.

        * http://uwsgi.readthedocs.io/en/latest/StaticFiles.html#caching-paths-mappings-resolutions

        :param int timeout: Amount of seconds to put resolved paths in the uWSGI cache.

        :param str|unicode cache_name: Cache name to use for static paths.

        """
        self._set('static-cache-paths', timeout)
        self._set('static-cache-paths-name', cache_name)

        return self._section