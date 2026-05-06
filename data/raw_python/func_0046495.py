def _build_cache_key(self, uri):
        """
        Build sha1 hex cache key to handle key length and whitespace to be compatible with Memcached
        """
        key = uri.clone(ext=None, version=None)

        if six.PY3:
            key = key.encode('utf-8')

        return sha1(key).hexdigest()