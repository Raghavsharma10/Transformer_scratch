def itemsize(self):
        """ Individual item sizes """
        return self._items[:self._count, 1] - self._items[:self._count, 0]