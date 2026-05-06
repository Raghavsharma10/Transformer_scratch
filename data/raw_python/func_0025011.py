def _initialize_uaa_cache(self):
        """
        If we don't yet have a uaa cache we need to
        initialize it.  As there may be more than one
        UAA instance we index by issuer and then store
        any clients, users, etc.
        """
        try:
            os.makedirs(os.path.dirname(self._cache_path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

        data = {}
        data[self.uri] = []

        return data