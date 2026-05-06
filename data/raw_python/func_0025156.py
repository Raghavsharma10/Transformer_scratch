def heapify(self, key=__marker):
        """
        Repair a broken heap. If the state of an item's priority value changes
        you can re-sort the relevant item only by providing ``key``.

        """
        if key is self.__marker:
            n = len(self._heap)
            for pos in reversed(range(n//2)):
                self._sink(pos)
        else:
            try:
                pos = self._position[key]
            except KeyError:
                raise KeyError(key)
            self._reheapify(pos)