def prefix_keys(self, prefix, strip_prefix=False):
        """Get all keys that begin with ``prefix``.

        :param prefix: Lexical prefix for keys to search.
        :type prefix: bytes

        :param strip_prefix: True to strip the prefix from yielded items.
        :type strip_prefix: bool

        :yields: All keys in the store that begin with ``prefix``.

        """
        keys = self.keys(key_from=prefix)

        start = 0
        if strip_prefix:
            start = len(prefix)

        for key in keys:
            if not key.startswith(prefix):
                break
            yield key[start:]