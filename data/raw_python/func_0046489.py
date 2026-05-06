def get(self, uri):
        """
        Return node for uri or None if not exists:
            {uri: x, content: y}
        """
        cache_key = self._build_cache_key(uri)
        value = self._get(cache_key)
        if value is not None:
            return self._decode_node(uri, value)