def find_le(self, k):
        'Return last item with a key <= k.  Raise ValueError if not found.'
        i = bisect_right(self._keys, k)
        if i:
            return self._items[i - 1]
        raise ValueError('No item found with key at or below: %r' % (k,))