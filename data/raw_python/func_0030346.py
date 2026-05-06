def find_ge_index(self, k):
        'Return first item with a key >= equal to k.  Raise ValueError if not found'
        i = bisect_left(self._keys, k)
        if i != len(self):
            return i
        raise ValueError('No item found with key at or above: %r' % (k,))