def get_many(self, uris):
        """
        Return request uri map of found nodes as dicts:
            {requested_uri: {uri: x, content: y}}
        """
        cache_keys = dict((self._build_cache_key(uri), uri) for uri in uris)
        result = self._get_many(cache_keys)
        nodes = {}
        for cache_key in result:
            uri = cache_keys[cache_key]
            value = result[cache_key]
            node = self._decode_node(uri, value)
            if node:
                nodes[uri] = node
        return nodes