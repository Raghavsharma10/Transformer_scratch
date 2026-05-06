def prefix_items(self, prefix, strip_prefix=False):
        """Get all (key, value) pairs with keys that begin with ``prefix``.

        :param prefix: Lexical prefix for keys to search.
        :type prefix: bytes

        :param strip_prefix: True to strip the prefix from yielded items.
        :type strip_prefix: bool

        :yields: All (key, value) pairs in the store where the keys
            begin with the ``prefix``.

        """
        items = self.items(key_from=prefix)

        start = 0
        if strip_prefix:
            start = len(prefix)

        for key, value in items:
            if not key.startswith(prefix):
                break
            yield key[start:], value