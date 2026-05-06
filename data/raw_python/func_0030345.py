def find_lt(self, k):
        """Return last item with a key < k.

        Raise ValueError if not found.

        """
        i = bisect_left(self._keys, k)
        if i:
            return self._items[i - 1]
        raise ValueError('No item found with key below: %r' % (k,))